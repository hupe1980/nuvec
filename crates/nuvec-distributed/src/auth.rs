//! Pre-shared-key authentication of a connection, before any data: both ends prove they
//! hold the key with HMAC-SHA256 over the mode, fresh random challenges from each side and
//! the two ranks, so a host without the key cannot join a group, a recorded handshake cannot
//! be replayed and the mode cannot be downgraded. When encrypting, each direction's key
//! comes from HKDF-SHA256 over the key, both challenges and the ranks (see [`crate::seal`]).
//!
//! The connecting end sends its mode's magic (`"nuvecENC"` encrypting, `"nuvecPSK"`
//! authenticating only), its challenge and its rank; the accepting end answers with its
//! challenge and its proof; the connecting end checks it and sends its own proof. Without a
//! key, the connecting end sends its rank alone.

use std::io::{Read, Write};
use std::net::TcpStream;

use hmac::{Hmac, Mac};
use nuvec_core::{Error, Result};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"nuvecPSK";
const MAGIC_ENC: &[u8; 8] = b"nuvecENC";
const SHORTEST: usize = 16;

/// A pre-shared key: SHA-256 of a domain tag and the secret. Never printed.
#[derive(Clone)]
pub(crate) struct Key([u8; 32]);

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key(..)")
    }
}

impl Key {
    /// The key of `secret`, at least 16 bytes (a recorded handshake allows offline guesses
    /// of a short one; `openssl rand -hex 32` makes a good secret).
    pub(crate) fn new(secret: &[u8]) -> Result<Self> {
        if secret.len() < SHORTEST {
            return Err(Error::InvalidArgument(format!(
                "distributed: the key must be at least {SHORTEST} bytes (e.g. `openssl rand -hex \
                 32`); it has {}",
                secret.len()
            )));
        }
        let mut h = Sha256::new();
        h.update(b"nuvec-distributed-key");
        h.update(secret);
        Ok(Self(h.finalize().into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn proof(
        &self,
        role: u8,
        encrypt: bool,
        first: &[u8; 32],
        second: &[u8; 32],
        c: u64,
        a: u64,
    ) -> Hmac<Sha256> {
        let mut m = Hmac::<Sha256>::new_from_slice(&self.0).expect("any key length");
        m.update(&[role, u8::from(encrypt)]);
        m.update(first);
        m.update(second);
        m.update(&c.to_le_bytes());
        m.update(&a.to_le_bytes());
        m
    }
}

/// HKDF-SHA256 (RFC 5869): `out.len()` bytes (at most 255 blocks) from `ikm`.
fn hkdf(salt: &[u8], ikm: &[u8], info: &[u8], out: &mut [u8]) {
    let mut extract = Hmac::<Sha256>::new_from_slice(salt).expect("any key length");
    extract.update(ikm);
    let prk = extract.finalize().into_bytes();
    let mut previous: Vec<u8> = Vec::new();
    for (i, block) in out.chunks_mut(32).enumerate() {
        let mut m = Hmac::<Sha256>::new_from_slice(&prk).expect("any key length");
        m.update(&previous);
        m.update(info);
        m.update(&[u8::try_from(i + 1).expect("at most 255 blocks")]);
        previous = m.finalize().into_bytes().to_vec();
        block.copy_from_slice(&previous[..block.len()]);
    }
}

/// The keys of an encrypted connection: what this end seals with and what it opens with,
/// each with its direction (`b'c'`: from the connecting end, `b'a'`: from the accepting end).
#[derive(Clone)]
pub(crate) struct Keys {
    pub(crate) send: ([u8; 32], u8),
    pub(crate) recv: ([u8; 32], u8),
}

impl std::fmt::Debug for Keys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Keys(..)")
    }
}

impl Keys {
    /// Both directions' keys of the connection from rank `c` (challenge `cc`) to rank `a`
    /// (challenge `ac`), as the connecting end (`connecting`) or the accepting one sees them.
    fn derive(key: &Key, cc: &[u8; 32], ac: &[u8; 32], c: u64, a: u64, connecting: bool) -> Self {
        let mut salt = [0u8; 64];
        salt[..32].copy_from_slice(cc);
        salt[32..].copy_from_slice(ac);
        let one = |dir: u8| {
            let mut info = b"nuvec-ring v1 ".to_vec();
            info.push(dir);
            info.extend_from_slice(&c.to_le_bytes());
            info.extend_from_slice(&a.to_le_bytes());
            let mut k = [0u8; 32];
            hkdf(&salt, &key.0, &info, &mut k);
            (k, dir)
        };
        let (from_c, from_a) = (one(b'c'), one(b'a'));
        match connecting {
            true => Self {
                send: from_c,
                recv: from_a,
            },
            false => Self {
                send: from_a,
                recv: from_c,
            },
        }
    }
}

fn magic(encrypt: bool) -> &'static [u8; 8] {
    if encrypt { MAGIC_ENC } else { MAGIC }
}

/// A secret, redacted in `Debug`.
#[derive(Clone, Default)]
pub(crate) struct Secret(pub(crate) Option<Vec<u8>>);

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Some(..)" } else { "None" })
    }
}

impl Secret {
    pub(crate) fn key(&self) -> Result<Option<Key>> {
        self.0.as_deref().map(Key::new).transpose()
    }
}

/// 32 bytes from the operating system's random source.
fn challenge() -> Result<[u8; 32]> {
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| Error::Backend(format!("distributed: reading /dev/urandom: {e}")))?;
    Ok(b)
}

fn io(peer: &str, e: std::io::Error) -> Error {
    Error::Backend(format!("distributed: {peer}: handshake: {e}"))
}

/// The connecting end: identifies as `rank` to `peer_rank` (named `peer` in errors), and
/// with `encrypt` returns the connection's keys.
pub(crate) fn connect(
    s: &TcpStream,
    key: Option<&Key>,
    encrypt: bool,
    rank: u64,
    peer_rank: u64,
    peer: &str,
) -> Result<Option<Keys>> {
    let mut s = s;
    let Some(key) = key else {
        s.write_all(&rank.to_le_bytes()).map_err(|e| io(peer, e))?;
        return Ok(None);
    };
    let mine = challenge()?;
    let mut hello = Vec::with_capacity(48);
    hello.extend_from_slice(magic(encrypt));
    hello.extend_from_slice(&mine);
    hello.extend_from_slice(&rank.to_le_bytes());
    s.write_all(&hello).map_err(|e| io(peer, e))?;
    let mut reply = [0u8; 64];
    s.read_exact(&mut reply).map_err(|e| {
        Error::Backend(format!(
            "distributed: {peer}: authentication failed (it has no key, another one, or the \
             other encryption setting): {e}"
        ))
    })?;
    let theirs: [u8; 32] = reply[..32].try_into().expect("32 bytes");
    key.proof(b'A', encrypt, &mine, &theirs, rank, peer_rank)
        .verify_slice(&reply[32..])
        .map_err(|_| {
            Error::Backend(format!(
                "distributed: {peer}: authentication failed (another key)"
            ))
        })?;
    let proof = key
        .proof(b'C', encrypt, &theirs, &mine, rank, peer_rank)
        .finalize()
        .into_bytes();
    s.write_all(&proof).map_err(|e| io(peer, e))?;
    Ok(encrypt.then(|| Keys::derive(key, &mine, &theirs, rank, peer_rank, true)))
}

/// The accepting end, rank `rank`: the connecting end's rank (checked against `expected`
/// when given), and with `encrypt` the connection's keys.
pub(crate) fn accept(
    s: &TcpStream,
    key: Option<&Key>,
    encrypt: bool,
    rank: u64,
    expected: Option<u64>,
    peer: &str,
) -> Result<(u64, Option<Keys>)> {
    let mut s = s;
    let mut first = [0u8; 8];
    s.read_exact(&mut first).map_err(|e| io(peer, e))?;
    let keyed = &first == MAGIC || &first == MAGIC_ENC;
    let check = |r: u64| match expected {
        Some(e) if e != r => Err(Error::Backend(format!(
            "distributed: {peer}: rank {r} connected instead of rank {e}"
        ))),
        _ => Ok(r),
    };
    let Some(key) = key else {
        if keyed {
            return Err(Error::Backend(format!(
                "distributed: {peer} uses a key (NUVEC_DISTRIBUTED_KEY) and this rank has none"
            )));
        }
        return Ok((check(u64::from_le_bytes(first))?, None));
    };
    if !keyed {
        return Err(Error::Backend(format!(
            "distributed: {peer} has no key (NUVEC_DISTRIBUTED_KEY) and this rank has one"
        )));
    }
    if &first != magic(encrypt) {
        let (theirs, ours) = match encrypt {
            true => ("does not encrypt", "does"),
            false => ("encrypts", "does not"),
        };
        return Err(Error::Backend(format!(
            "distributed: {peer} {theirs} (NUVEC_DISTRIBUTED_ENCRYPT) and this rank {ours}"
        )));
    }
    let mut hello = [0u8; 40];
    s.read_exact(&mut hello).map_err(|e| io(peer, e))?;
    let theirs: [u8; 32] = hello[..32].try_into().expect("32 bytes");
    let peer_rank = check(u64::from_le_bytes(hello[32..].try_into().expect("8 bytes")))?;
    let mine = challenge()?;
    let mut reply = Vec::with_capacity(64);
    reply.extend_from_slice(&mine);
    reply.extend_from_slice(
        &key.proof(b'A', encrypt, &theirs, &mine, peer_rank, rank)
            .finalize()
            .into_bytes(),
    );
    s.write_all(&reply).map_err(|e| io(peer, e))?;
    let mut proof = [0u8; 32];
    s.read_exact(&mut proof).map_err(|e| {
        Error::Backend(format!(
            "distributed: {peer}: authentication failed (it refused this rank's proof: another \
             key): {e}"
        ))
    })?;
    key.proof(b'C', encrypt, &mine, &theirs, peer_rank, rank)
        .verify_slice(&proof)
        .map_err(|_| {
            Error::Backend(format!(
                "distributed: {peer}: authentication failed (another key)"
            ))
        })?;
    let keys = encrypt.then(|| Keys::derive(key, &theirs, &mine, peer_rank, rank, false));
    Ok((peer_rank, keys))
}

#[cfg(test)]
mod tests {
    use std::net::{TcpListener, TcpStream};

    use super::*;

    /// A connected pair with the read timeouts real connections have.
    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let a = l.accept().unwrap().0;
        for s in [&c, &a] {
            s.set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
        }
        (c, a)
    }

    type Ends = (Result<Option<Keys>>, Result<(u64, Option<Keys>)>);

    fn run_with(ck: Option<&str>, ak: Option<&str>, ce: bool, ae: bool) -> Ends {
        let (c, a) = pair();
        let key = |s: Option<&str>| s.map(|s| Key::new(s.as_bytes()).unwrap());
        let (ck, ak) = (key(ck), key(ak));
        std::thread::scope(|s| {
            let h = s.spawn(|| connect(&c, ck.as_ref(), ce, 3, 4, "rank 4"));
            let got = accept(&a, ak.as_ref(), ae, 4, Some(3), "rank 3");
            // A failed acceptor closes its end, so the connector's read ends.
            drop(a);
            (h.join().unwrap(), got)
        })
    }

    fn run(ck: Option<&str>, ak: Option<&str>) -> (Result<Option<Keys>>, Result<u64>) {
        let (c, a) = run_with(ck, ak, false, false);
        (c, a.map(|(r, _)| r))
    }

    const K: &str = "0123456789abcdef0123";

    #[test]
    fn keys_authenticate_both_ends() {
        let (c, a) = run(Some(K), Some(K));
        assert!(c.is_ok() && a.unwrap() == 3);
        let (c, a) = run(None, None);
        assert!(c.is_ok() && a.unwrap() == 3);
        let (c, a) = run(Some(K), Some("another key, long enough"));
        let e = c.unwrap_err().to_string();
        assert!(e.contains("authentication failed") && !e.contains(K), "{e}");
        assert!(a.is_err());
        let e = run(None, Some(K)).1.unwrap_err().to_string();
        assert!(e.contains("has no key"), "{e}");
        let e = run(Some(K), None).1.unwrap_err().to_string();
        assert!(e.contains("uses a key"), "{e}");
        let e = Key::new(b"short").err().unwrap().to_string();
        assert!(e.contains("at least 16 bytes"), "{e}");
        assert_eq!(format!("{:?}", Key::new(K.as_bytes()).unwrap()), "Key(..)");
        assert_eq!(format!("{:?}", Secret(Some(K.into()))), "Some(..)");
    }

    #[test]
    fn a_replayed_handshake_is_refused() {
        // A recorded connector's messages fail against a fresh challenge.
        let key = Key::new(K.as_bytes()).unwrap();
        let (c, a) = pair();
        let mut c2 = &c;
        let old = [7u8; 32];
        let mut hello = MAGIC.to_vec();
        hello.extend_from_slice(&old);
        hello.extend_from_slice(&3u64.to_le_bytes());
        c2.write_all(&hello).unwrap();
        // The proof a past acceptor challenge would have taken.
        let stale = key
            .proof(b'C', false, &[9u8; 32], &old, 3, 4)
            .finalize()
            .into_bytes();
        let got = std::thread::scope(|s| {
            let h = s.spawn(|| accept(&a, Some(&key), false, 4, Some(3), "rank 3"));
            let mut reply = [0u8; 64];
            c2.read_exact(&mut reply).unwrap();
            c2.write_all(&stale).unwrap();
            h.join().unwrap()
        });
        assert!(
            got.map(|(r, _)| r)
                .unwrap_err()
                .to_string()
                .contains("authentication failed")
        );
    }

    #[test]
    fn encrypting_ends_agree_on_keys_and_refuse_the_other_mode() {
        let (c, a) = run_with(Some(K), Some(K), true, true);
        let (c, (rank, a)) = (c.unwrap().unwrap(), a.unwrap());
        let a = a.unwrap();
        assert_eq!(rank, 3);
        assert_eq!((c.send, c.recv), (a.recv, a.send));
        assert_ne!(c.send.0, c.recv.0);
        assert_eq!((c.send.1, c.recv.1), (b'c', b'a'));
        // Fresh challenges: another connection, other keys.
        let (c2, _) = run_with(Some(K), Some(K), true, true);
        assert_ne!(c2.unwrap().unwrap().send.0, c.send.0);
        let e = run_with(Some(K), Some(K), false, true)
            .1
            .unwrap_err()
            .to_string();
        assert!(e.contains("rank 3 does not encrypt"), "{e}");
        let e = run_with(Some(K), Some(K), true, false)
            .1
            .unwrap_err()
            .to_string();
        assert!(e.contains("rank 3 encrypts"), "{e}");
        assert_eq!(format!("{c:?}"), "Keys(..)");
    }

    #[test]
    fn hkdf_sha256_matches_rfc_5869() {
        // RFC 5869 test case 1.
        let hex = |s: &str| {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect::<Vec<u8>>()
        };
        let mut okm = [0u8; 42];
        hkdf(
            &hex("000102030405060708090a0b0c"),
            &[0x0b; 22],
            &hex("f0f1f2f3f4f5f6f7f8f9"),
            &mut okm,
        );
        let want =
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865";
        assert_eq!(okm.to_vec(), hex(want));
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231() {
        // RFC 4231 test case 2.
        let mut m = Hmac::<Sha256>::new_from_slice(b"Jefe").unwrap();
        m.update(b"what do ya want for nothing?");
        let want = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        let got: String = m
            .finalize()
            .into_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(got, want);
    }
}
