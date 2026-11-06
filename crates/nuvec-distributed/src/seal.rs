//! Encrypted ring connections: every frame sealed with ChaCha20-Poly1305 under the key of
//! its direction (from the handshake, [`crate::auth::Keys`]).
//!
//! A frame is a 12-byte header (the plaintext length, `u32`, and the frame's counter, `u64`,
//! little-endian), the ciphertext and a 16-byte tag; the header is the associated data and
//! the nonce is the direction followed by the counter. The receiver expects the counters in
//! order, so a replayed, reordered or dropped frame fails with its own error, a changed one
//! fails its tag, and no counter is used twice. Framing does not depend on how a transfer is
//! split into calls: a frame's plaintext left over from one read is returned by the next.

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::{Mutex, MutexGuard, PoisonError};

use chacha20poly1305::aead::AeadInPlace;
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce, Tag};

use crate::auth::Keys;

/// Plaintext bytes per frame, at most.
pub(crate) const FRAME: usize = 64 << 10;
const HEADER: usize = 12;
const TAG: usize = 16;

/// What a read or write moved: bytes on the wire, and plaintext bytes delivered (read) or
/// fully sent (write), or the peer closed the connection.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Moved {
    Bytes { wire: usize, plain: usize },
    Closed,
}

fn invalid(msg: String) -> std::io::Error {
    std::io::Error::new(ErrorKind::InvalidData, msg)
}

fn nonce(dir: u8, counter: u64) -> Nonce {
    let mut n = [0u8; 12];
    n[0] = dir;
    n[4..].copy_from_slice(&counter.to_le_bytes());
    n.into()
}

struct Sealer {
    cipher: ChaCha20Poly1305,
    dir: u8,
    counter: u64,
    /// The sealed frames being written, and how much of them is on the wire.
    frames: Vec<u8>,
    at: usize,
    /// The plaintext length of `frames`.
    plain: usize,
}

/// Frames sealed or opened together, on several threads (ChaCha20-Poly1305 runs at about
/// 0.7 GB/s on one core).
const BATCH: usize = 16;

/// One frame of a batch: its bytes (header, body and tag), and whether it sealed or opened.
struct Part<'a> {
    bytes: &'a mut [u8],
    counter: u64,
    ok: bool,
}

/// `bytes` cut into frames of the given total lengths.
fn parts<'a>(mut bytes: &'a mut [u8], lens: &[usize], counter: u64) -> Vec<Part<'a>> {
    let mut out = Vec::with_capacity(lens.len());
    for (i, &len) in lens.iter().enumerate() {
        let (frame, rest) = std::mem::take(&mut bytes).split_at_mut(len);
        bytes = rest;
        out.push(Part {
            bytes: frame,
            counter: counter + i as u64,
            ok: false,
        });
    }
    out
}

impl Sealer {
    /// Seals up to [`BATCH`] frames of [`FRAME`] bytes (at most) from `data` into
    /// `self.frames`.
    fn seal(&mut self, data: &[u8]) -> std::io::Result<()> {
        let n = data.len().min(BATCH * FRAME);
        let count = n.div_ceil(FRAME);
        if self.counter.checked_add(count as u64).is_none() {
            return Err(invalid("the frame counter reached its limit".into()));
        }
        self.frames.clear();
        let mut lens = Vec::with_capacity(count);
        for (i, chunk) in data[..n].chunks(FRAME).enumerate() {
            self.frames.extend_from_slice(
                &u32::try_from(chunk.len())
                    .expect("FRAME fits")
                    .to_le_bytes(),
            );
            self.frames
                .extend_from_slice(&(self.counter + i as u64).to_le_bytes());
            self.frames.extend_from_slice(chunk);
            self.frames.extend_from_slice(&[0; TAG]);
            lens.push(HEADER + chunk.len() + TAG);
        }
        let (cipher, dir) = (&self.cipher, self.dir);
        let mut batch = parts(&mut self.frames, &lens, self.counter);
        crate::pool::for_each(&mut batch, &|p: &mut Part<'_>| {
            let end = p.bytes.len() - TAG;
            let (header, rest) = p.bytes.split_at_mut(HEADER);
            let (body, tag) = rest.split_at_mut(end - HEADER);
            if let Ok(t) = cipher.encrypt_in_place_detached(&nonce(dir, p.counter), header, body) {
                tag.copy_from_slice(&t);
                p.ok = true;
            }
        });
        if batch.iter().any(|p| !p.ok) {
            return Err(invalid("sealing a frame failed".into()));
        }
        self.counter += count as u64;
        (self.at, self.plain) = (0, n);
        Ok(())
    }
}

struct Opener {
    cipher: ChaCha20Poly1305,
    dir: u8,
    counter: u64,
    /// Bytes received: `have` of them, the first `parsed` in opened frames whose plaintext
    /// not yet returned is `ready`, then the next frames' bytes.
    buf: Vec<u8>,
    have: usize,
    parsed: usize,
    ready: std::collections::VecDeque<std::ops::Range<usize>>,
}

impl Opener {
    /// The length of the frame at `at` once its header is in (checked: its size, and its
    /// counter `counter`).
    fn frame_len(&self, at: usize, counter: u64) -> std::io::Result<Option<usize>> {
        if self.have - at < HEADER {
            return Ok(None);
        }
        let h = &self.buf[at..at + HEADER];
        let len = u32::from_le_bytes(h[..4].try_into().expect("4 bytes")) as usize;
        if len > FRAME {
            return Err(invalid(format!(
                "a frame of {len} bytes (at most {FRAME}): not from this connection"
            )));
        }
        let got = u64::from_le_bytes(h[4..].try_into().expect("8 bytes"));
        if got != counter {
            return Err(invalid(format!(
                "frame {got} arrived where frame {counter} was due: replayed, reordered or \
                 dropped"
            )));
        }
        Ok(Some(HEADER + len + TAG))
    }

    /// Opens every complete frame received (on several threads); their plaintext becomes
    /// `ready`. Checks the header of an incomplete last frame too.
    fn open(&mut self) -> std::io::Result<()> {
        let (mut at, mut lens) = (self.parsed, Vec::new());
        while let Some(len) = self.frame_len(at, self.counter + lens.len() as u64)? {
            if self.have - at < len {
                break;
            }
            lens.push(len);
            at += len;
        }
        if lens.is_empty() {
            return Ok(());
        }
        let (cipher, dir) = (&self.cipher, self.dir);
        let mut batch = parts(&mut self.buf[self.parsed..at], &lens, self.counter);
        crate::pool::for_each(&mut batch, &|p: &mut Part<'_>| {
            let end = p.bytes.len() - TAG;
            let (head, tag) = p.bytes.split_at_mut(end);
            let tag = Tag::clone_from_slice(tag);
            let (header, body) = head.split_at_mut(HEADER);
            p.ok = cipher
                .decrypt_in_place_detached(&nonce(dir, p.counter), header, body, &tag)
                .is_ok();
        });
        if batch.iter().any(|p| !p.ok) {
            return Err(invalid(
                "a frame failed authentication: changed in transit, or another key".into(),
            ));
        }
        let mut from = self.parsed;
        for len in &lens {
            self.ready.push_back(from + HEADER..from + len - TAG);
            from += len;
        }
        self.counter += lens.len() as u64;
        self.parsed = at;
        Ok(())
    }
}

/// A ring connection: the socket, and when encrypting, its sealing and opening states (each
/// used by one transfer at a time; the locks make the link shareable).
pub(crate) struct Link {
    stream: TcpStream,
    seal: Option<Mutex<Sealer>>,
    open: Option<Mutex<Opener>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Link {
    /// `stream`, encrypted with `keys` when given.
    pub(crate) fn new(stream: TcpStream, keys: Option<Keys>) -> Self {
        let Some(keys) = keys else {
            return Self {
                stream,
                seal: None,
                open: None,
            };
        };
        let cipher = |k: &[u8; 32]| ChaCha20Poly1305::new(k.into());
        Self {
            stream,
            seal: Some(Mutex::new(Sealer {
                cipher: cipher(&keys.send.0),
                dir: keys.send.1,
                counter: 0,
                frames: Vec::with_capacity(BATCH * (HEADER + FRAME + TAG)),
                at: 0,
                plain: 0,
            })),
            open: Some(Mutex::new(Opener {
                cipher: cipher(&keys.recv.0),
                dir: keys.recv.1,
                counter: 0,
                buf: vec![0; BATCH * (HEADER + FRAME + TAG)],
                have: 0,
                parsed: 0,
                ready: std::collections::VecDeque::new(),
            })),
        }
    }

    pub(crate) fn stream(&self) -> &TcpStream {
        &self.stream
    }

    /// Whether received plaintext is waiting, so a read returns without the socket.
    pub(crate) fn buffered(&self) -> bool {
        self.open
            .as_ref()
            .is_some_and(|o| !lock(o).ready.is_empty())
    }

    /// Writes from `data` (non-empty; the same bytes again after a write that sent no
    /// whole frame). Errors as the socket's: `WouldBlock` when nothing moved.
    pub(crate) fn write(&self, data: &[u8]) -> std::io::Result<Moved> {
        let Some(seal) = &self.seal else {
            return Ok(match (&self.stream).write(data)? {
                0 => Moved::Closed,
                n => Moved::Bytes { wire: n, plain: n },
            });
        };
        let mut s = lock(seal);
        if s.frames.is_empty() {
            s.seal(data)?;
        }
        let mut wire = 0;
        while s.at < s.frames.len() {
            let at = s.at;
            match (&self.stream).write(&s.frames[at..]) {
                Ok(0) => return Ok(Moved::Closed),
                Ok(n) => {
                    s.at += n;
                    wire += n;
                }
                Err(e) if wire > 0 && e.kind() == ErrorKind::WouldBlock => {
                    return Ok(Moved::Bytes { wire, plain: 0 });
                }
                Err(e) => return Err(e),
            }
        }
        s.frames.clear();
        Ok(Moved::Bytes {
            wire,
            plain: s.plain,
        })
    }

    /// Reads into `out` (non-empty). Errors as the socket's: `WouldBlock` when nothing
    /// moved and nothing was waiting.
    pub(crate) fn read(&self, out: &mut [u8]) -> std::io::Result<Moved> {
        let Some(open) = &self.open else {
            return Ok(match (&self.stream).read(out)? {
                0 => Moved::Closed,
                n => Moved::Bytes { wire: n, plain: n },
            });
        };
        let mut o = lock(open);
        let mut wire = 0;
        while o.ready.is_empty() {
            // Opened frames all returned: their bytes go, the rest moves to the front.
            if o.parsed > 0 {
                let (parsed, have) = (o.parsed, o.have);
                o.buf.copy_within(parsed..have, 0);
                (o.have, o.parsed) = (have - parsed, 0);
            }
            let have = o.have;
            match (&self.stream).read(&mut o.buf[have..]) {
                Ok(0) => return Ok(Moved::Closed),
                Ok(n) => {
                    o.have += n;
                    wire += n;
                    o.open()?;
                }
                Err(e) if wire > 0 && e.kind() == ErrorKind::WouldBlock => {
                    return Ok(Moved::Bytes { wire, plain: 0 });
                }
                Err(e) => return Err(e),
            }
        }
        let o: &mut Opener = &mut o;
        let mut n = 0;
        while n < out.len() {
            let Some(r) = o.ready.front_mut() else { break };
            let k = r.len().min(out.len() - n);
            out[n..n + k].copy_from_slice(&o.buf[r.start..r.start + k]);
            r.start += k;
            n += k;
            if r.start == r.end {
                o.ready.pop_front();
            }
        }
        Ok(Moved::Bytes { wire, plain: n })
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    fn keys(dir: u8) -> Keys {
        Keys {
            send: ([7; 32], dir),
            recv: ([7; 32], dir),
        }
    }

    /// A connected, non-blocking pair: the first end seals with direction `c`, the second
    /// opens it.
    fn pair() -> (Link, Link, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let a = l.accept().unwrap().0;
        let raw = c.try_clone().unwrap();
        for s in [&c, &a] {
            s.set_nonblocking(true).unwrap();
        }
        (
            Link::new(c, Some(keys(b'c'))),
            Link::new(a, Some(keys(b'c'))),
            raw,
        )
    }

    /// Moves `data` from `tx` to `rx`, reading in pieces of `piece` bytes.
    fn transfer(tx: &Link, rx: &Link, data: &[u8], piece: usize) -> std::io::Result<Vec<u8>> {
        let (mut sent, mut got) = (0, Vec::new());
        let mut buf = vec![0u8; piece];
        while got.len() < data.len() {
            if sent < data.len() {
                match tx.write(&data[sent..]) {
                    Ok(Moved::Bytes { plain, .. }) => sent += plain,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                    other => panic!("{other:?}"),
                }
            }
            let want = piece.min(data.len() - got.len());
            match rx.read(&mut buf[..want]) {
                Ok(Moved::Bytes { plain, .. }) => got.extend_from_slice(&buf[..plain]),
                Ok(Moved::Closed) => panic!("closed"),
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => return Err(e),
            }
        }
        Ok(got)
    }

    #[test]
    fn frames_carry_any_split_of_a_transfer() {
        let data: Vec<u8> = (0..(BATCH + 3) * FRAME + 1234)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        for piece in [1, 7, 4096, FRAME, 5 * FRAME] {
            let (tx, rx, _) = pair();
            let got = transfer(&tx, &rx, &data, piece).unwrap();
            assert!(got == data, "piece {piece}");
            assert!(!rx.buffered());
        }
    }

    /// A sealed frame of `data` as the sender's counter `counter` would write it.
    fn frame(data: &[u8], counter: u64) -> Vec<u8> {
        let mut s = Sealer {
            cipher: ChaCha20Poly1305::new((&[7u8; 32]).into()),
            dir: b'c',
            counter,
            frames: Vec::new(),
            at: 0,
            plain: 0,
        };
        s.seal(data).unwrap();
        s.frames
    }

    /// Receives `bytes` as written to the socket: the first error, or `"opened"` once every
    /// frame opened.
    fn receive(bytes: &[u8]) -> String {
        let (_tx, rx, mut raw) = pair();
        raw.write_all(bytes).unwrap();
        let mut out = vec![0u8; 64];
        let (start, mut wired) = (std::time::Instant::now(), 0);
        loop {
            match rx.read(&mut out) {
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    assert!(start.elapsed().as_secs() < 5, "nothing arrived");
                }
                Err(e) => return e.to_string(),
                Ok(Moved::Bytes { wire, .. }) => wired += wire,
                Ok(Moved::Closed) => panic!("closed"),
            }
            if wired == bytes.len() && !rx.buffered() {
                return "opened".into();
            }
        }
    }

    #[test]
    fn changed_replayed_reordered_and_oversized_frames_are_refused() {
        let first = frame(b"hello", 0);
        assert_eq!(receive(&first), "opened");
        let mut changed = first.clone();
        changed[HEADER + 1] ^= 1;
        assert!(receive(&changed).contains("failed authentication"));
        let mut tag = first.clone();
        *tag.last_mut().unwrap() ^= 1;
        assert!(receive(&tag).contains("failed authentication"));
        // The same frame twice: the second is a replay.
        let e = receive(&[first.clone(), first.clone()].concat());
        assert!(e.contains("frame 0 arrived where frame 1 was due"), "{e}");
        // Frame 1 first: reordered (or 0 dropped).
        let e = receive(&frame(b"world", 1));
        assert!(e.contains("frame 1 arrived where frame 0 was due"), "{e}");
        // A changed counter in the header (still in order for the receiver) fails the tag.
        let mut moved = frame(b"world", 1);
        moved[4] = 0;
        assert!(receive(&moved).contains("failed authentication"));
        let mut big = first;
        big[..4].copy_from_slice(&(FRAME as u32 + 1).to_le_bytes());
        assert!(receive(&big).contains("at most"));
    }

    #[test]
    fn a_counter_never_repeats() {
        let mut s = Sealer {
            cipher: ChaCha20Poly1305::new((&[7u8; 32]).into()),
            dir: b'c',
            counter: u64::MAX,
            frames: Vec::new(),
            at: 0,
            plain: 0,
        };
        assert!(s.seal(b"x").unwrap_err().to_string().contains("limit"));
    }

    #[test]
    fn chacha20_poly1305_matches_rfc_8439() {
        // RFC 8439 §2.8.2.
        let hex = |s: &str| {
            let s: String = s.split_whitespace().collect();
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect::<Vec<u8>>()
        };
        let key: [u8; 32] = std::array::from_fn(|i| 0x80 + i as u8);
        let nonce = hex("07000000 4041424344454647");
        let aad = hex("50515253c0c1c2c3c4c5c6c7");
        let mut text = b"Ladies and Gentlemen of the class of '99: If I could offer you only one \
                         tip for the future, sunscreen would be it."
            .to_vec();
        let tag = ChaCha20Poly1305::new((&key).into())
            .encrypt_in_place_detached(Nonce::from_slice(&nonce), &aad, &mut text)
            .unwrap();
        assert_eq!(text[..16].to_vec(), hex("d31a8d34648e60db7b86afbc53ef7ec2"));
        assert_eq!(tag.to_vec(), hex("1ae10b594f09e26a7e902ecbd0600691"));
    }
}
