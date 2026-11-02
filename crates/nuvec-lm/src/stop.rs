//! Where generation stops besides end-of-sequence and `max_tokens` (vLLM's `stop`,
//! `stop_token_ids`): stop tokens, which end it like end-of-sequence and are not emitted,
//! and stop strings, matched on the generated text's bytes as the tokenizer decodes them.
//!
//! A stop string ends generation at the token that completes it. That token is emitted,
//! since tokens are atomic, and [`Stopped::String`] says how many bytes to cut from the
//! end of the decoded output so that the text ends before the stop string.
//! [`TextStream`] applies the cut while streaming: it holds back any text that could begin
//! a stop string, so a stop string never shows, even in part.

use std::sync::Arc;

use nuvec::{Error, Result};

use crate::GenerateOptions;
use crate::tokenizer::Tokenizer;

/// The most stop strings a request may have.
pub const MAX_STOP_STRINGS: usize = 64;
/// The longest stop string, in bytes.
pub const MAX_STOP_BYTES: usize = 1024;

/// Each vocabulary token's bytes, and how many leading spaces the decoder strips.
#[derive(Debug)]
struct TokenBytes {
    bytes: Vec<Box<[u8]>>,
    strip: usize,
}

/// Stop tokens and stop strings ([module documentation](self)). The default stops nowhere.
#[derive(Debug, Clone, Default)]
pub struct Stop {
    tokens: Vec<u32>,
    strings: Vec<Box<[u8]>>,
    /// The tokenizer's bytes, shared by every request built from the same `Stop`.
    table: Option<Arc<TokenBytes>>,
}

impl PartialEq for Stop {
    fn eq(&self, other: &Self) -> bool {
        let tables = match (&self.table, &other.table) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (a, b) => a.is_none() && b.is_none(),
        };
        self.tokens == other.tokens && self.strings == other.strings && tables
    }
}

impl Stop {
    /// Stops at any of `ids` (not emitted, like end-of-sequence).
    pub fn tokens(ids: impl IntoIterator<Item = u32>) -> Self {
        Self::default().with_tokens(ids)
    }

    /// Stops after the text `tokenizer` decodes contains any of `strings`. Each string must
    /// be non-empty and at most [`MAX_STOP_BYTES`] bytes, with at most
    /// [`MAX_STOP_STRINGS`] of them. Building one reads every token's bytes once; clone the
    /// result for each request.
    pub fn strings<S: AsRef<str>>(
        tokenizer: &Tokenizer,
        strings: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let strings: Vec<Box<[u8]>> = strings
            .into_iter()
            .map(|s| s.as_ref().as_bytes().into())
            .collect();
        let invalid = |msg: String| Err(Error::InvalidArgument(format!("stop strings: {msg}")));
        if strings.len() > MAX_STOP_STRINGS {
            return invalid(format!(
                "{} strings, at most {MAX_STOP_STRINGS}",
                strings.len()
            ));
        }
        if let Some(i) = strings.iter().position(|s| s.is_empty()) {
            return invalid(format!("string {i} is empty"));
        }
        if let Some(i) = strings.iter().position(|s| s.len() > MAX_STOP_BYTES) {
            return invalid(format!("string {i} is longer than {MAX_STOP_BYTES} bytes"));
        }
        let bytes = (0..tokenizer.vocab_size() as u32)
            .map(|id| tokenizer.token_bytes(&[id]).into_boxed_slice())
            .collect();
        Ok(Self {
            tokens: Vec::new(),
            strings,
            table: Some(Arc::new(TokenBytes {
                bytes,
                strip: tokenizer.strip_count(),
            })),
        })
    }

    /// These stops and also `ids`.
    pub fn with_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        for id in ids {
            if !self.tokens.contains(&id) {
                self.tokens.push(id);
            }
        }
        self
    }

    /// The stop tokens.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Whether it stops nowhere.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.strings.is_empty()
    }

    /// The longest stop string's length (0 without any).
    fn longest(&self) -> usize {
        self.strings.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}

/// Which stop ended a generation ([`crate::FinishReason::Stop`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stopped {
    /// A stop token (not emitted).
    Token(u32),
    /// The stop string at `index`. The token that completed it was emitted; the last
    /// `cut` bytes of the decoded output are the stop string and anything after it in
    /// that token, so the text before the stop string is the output minus `cut` bytes.
    String {
        /// Its index among the stop strings.
        index: usize,
        /// The bytes to cut from the end of the decoded output.
        cut: usize,
    },
}

/// The leading spaces a SentencePiece-style decoder strips from the start of a text, as
/// [`Tokenizer::decode_bytes`] and [`crate::StreamDecoder`] strip them.
#[derive(Debug, Clone)]
struct Strip(Option<usize>);

impl Default for Strip {
    fn default() -> Self {
        Self(Some(0))
    }
}

impl Strip {
    fn apply(&mut self, strip: usize, bytes: &mut Vec<u8>) {
        if let Some(done) = self.0 {
            let limit = strip.saturating_sub(done);
            let n = bytes.iter().take(limit).take_while(|&&b| b == b' ').count();
            bytes.drain(..n);
            self.0 = bytes.is_empty().then_some(done + n);
        }
    }
}

/// A request's progress through its stop strings: the last bytes of its text, enough for
/// a match that began in earlier tokens.
#[derive(Debug, Clone, Default)]
pub(crate) struct StopState {
    tail: Vec<u8>,
    strip: Strip,
}

impl StopState {
    /// The stop `token`, the `n`-th generated (from 1), hits. Stop strings count from the
    /// `min_tokens`-th token on (vLLM's rule); a match that began earlier counts if it
    /// completes there.
    pub(crate) fn push(
        &mut self,
        stop: &Stop,
        token: u32,
        n: usize,
        min_tokens: usize,
    ) -> Option<Stopped> {
        if stop.tokens.contains(&token) {
            return Some(Stopped::Token(token));
        }
        let table = stop.table.as_ref()?;
        let mut bytes = table
            .bytes
            .get(token as usize)
            .map_or_else(Vec::new, |b| b.to_vec());
        self.strip.apply(table.strip, &mut bytes);
        self.find(stop, &bytes, n >= min_tokens)
            .map(|(index, cut)| Stopped::String { index, cut })
    }

    /// Appends `bytes` and, when `check`, finds the earliest-starting stop string that
    /// ends inside them (ties: the lowest index), as `(index, cut)`.
    fn find(&mut self, stop: &Stop, bytes: &[u8], check: bool) -> Option<(usize, usize)> {
        if bytes.is_empty() {
            return None;
        }
        let old = self.tail.len();
        self.tail.extend_from_slice(bytes);
        let window = &self.tail;
        let mut best: Option<(usize, usize)> = None;
        if check {
            for (index, s) in stop.strings.iter().enumerate() {
                // Matches that end inside the new bytes start after `old - len`.
                let first = (old + 1).saturating_sub(s.len());
                let Some(last) = window.len().checked_sub(s.len()) else {
                    continue;
                };
                if let Some(start) = (first..=last).find(|&i| window[i..].starts_with(s))
                    && best.is_none_or(|(_, b)| start < b)
                {
                    best = Some((index, start));
                }
            }
        }
        let found = best.map(|(index, start)| (index, window.len() - start));
        // Keep only what a later match can start in.
        let keep = stop.longest().saturating_sub(1);
        let drop = self.tail.len().saturating_sub(keep);
        self.tail.drain(..drop);
        found
    }
}

/// Text of a token stream that never shows a stop string: [`crate::StreamDecoder`] that
/// also holds back any ending that could begin one of the request's stop strings, and cuts
/// the text at the first one. Push the tokens a generation emits, then [`Self::finish`].
#[derive(Debug, Clone)]
pub struct TextStream<'t> {
    tokenizer: &'t Tokenizer,
    stop: Stop,
    min_tokens: usize,
    state: StopState,
    strip: Strip,
    pending: Vec<u8>,
    produced: usize,
    done: bool,
}

impl<'t> TextStream<'t> {
    /// A stream for a generation with `options` (its stop strings and `min_tokens`).
    pub fn new(tokenizer: &'t Tokenizer, options: &GenerateOptions) -> Self {
        Self {
            tokenizer,
            stop: options.stop.clone(),
            min_tokens: options.min_tokens,
            state: StopState::default(),
            strip: Strip::default(),
            pending: Vec::new(),
            produced: 0,
            done: false,
        }
    }

    /// The text that `id`, the next emitted token, makes safe to show: whole characters
    /// that cannot begin a stop string. After a stop string, nothing.
    pub fn push(&mut self, id: u32) -> String {
        if self.done {
            return String::new();
        }
        self.produced += 1;
        let mut bytes = self.tokenizer.token_bytes(&[id]);
        self.strip.apply(self.tokenizer.strip_count(), &mut bytes);
        let found = self
            .state
            .find(&self.stop, &bytes, self.produced >= self.min_tokens);
        self.pending.extend_from_slice(&bytes);
        if let Some((_, cut)) = found {
            // The match begins in held-back bytes: nothing of it was shown.
            let keep = self.pending.len() - cut;
            self.pending.truncate(keep);
            self.done = true;
            return take_text(&mut self.pending, keep, true);
        }
        let hold = self
            .stop
            .strings
            .iter()
            .map(|s| {
                (1..s.len().min(self.pending.len() + 1))
                    .rev()
                    .find(|&k| self.pending.ends_with(&s[..k]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let upto = self.pending.len() - hold;
        take_text(&mut self.pending, upto, false)
    }

    /// The text still held back (cut at a stop string if one matched).
    pub fn finish(&mut self) -> String {
        let n = self.pending.len();
        take_text(&mut self.pending, n, true)
    }
}

/// Removes and returns the text of `bytes[..upto]`: invalid sequences as U+FFFD, and,
/// unless `all`, an unfinished character at the end left in `bytes`.
fn take_text(bytes: &mut Vec<u8>, upto: usize, all: bool) -> String {
    let mut text = String::new();
    let mut done = 0;
    while done < upto {
        match std::str::from_utf8(&bytes[done..upto]) {
            Ok(s) => {
                text.push_str(s);
                done = upto;
            }
            Err(e) => {
                let valid = done + e.valid_up_to();
                text.push_str(std::str::from_utf8(&bytes[done..valid]).expect("valid up to"));
                done = valid;
                match e.error_len() {
                    Some(n) => {
                        text.push(char::REPLACEMENT_CHARACTER);
                        done += n;
                    }
                    None if all => {
                        text.push(char::REPLACEMENT_CHARACTER);
                        done = upto;
                    }
                    None => break,
                }
            }
        }
    }
    bytes.drain(..done);
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A byte-level stand-in for a tokenizer: token `i` is `vocab[i]`.
    fn stop_over(vocab: &[&[u8]], strings: &[&str], strip: usize) -> Stop {
        Stop {
            tokens: Vec::new(),
            strings: strings.iter().map(|s| s.as_bytes().into()).collect(),
            table: Some(Arc::new(TokenBytes {
                bytes: vocab.iter().map(|&b| b.into()).collect(),
                strip,
            })),
        }
    }

    /// The first stop string in `text` (bytes), whole-text reference: `(index, start)` of
    /// the earliest start among strings ending at the first possible end.
    fn reference(text: &[u8], strings: &[&str]) -> Option<(usize, usize)> {
        (1..=text.len())
            .find_map(|end| {
                strings
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        let s = s.as_bytes();
                        (s.len() <= end && &text[end - s.len()..end] == s)
                            .then(|| (i, end - s.len()))
                    })
                    .min_by_key(|&(i, start)| (start, i))
                    .map(|(i, start)| (i, start, end))
            })
            .map(|(i, start, _)| (i, start))
    }

    fn run(stop: &Stop, tokens: &[u32], min: usize) -> Option<(usize, Stopped)> {
        let mut s = StopState::default();
        tokens
            .iter()
            .enumerate()
            .find_map(|(i, &t)| s.push(stop, t, i + 1, min).map(|m| (i, m)))
    }

    #[test]
    fn a_stop_string_across_tokens_and_split_characters() {
        // "ab", "c\n", "\nd", "é" split in its two bytes.
        let vocab: &[&[u8]] = &[b"ab", b"c\n", b"\nd", &[0xC3], &[0xA9], b"x"];
        let stop = stop_over(vocab, &["\n\n", "é"], 0);
        // "ab" "c\n" "\nd": the match "\n\n" completes in token 2 (0-based), 2 bytes
        // before its end ("\nd" holds "\n" of it and "d").
        assert_eq!(
            run(&stop, &[0, 1, 2], 0),
            Some((2, Stopped::String { index: 0, cut: 3 }))
        );
        // "é" arrives in two tokens: the match completes at the second.
        assert_eq!(
            run(&stop, &[5, 3, 4, 5], 0),
            Some((2, Stopped::String { index: 1, cut: 2 }))
        );
        // `min_tokens` defers matching to the third token; a match completing there that
        // began earlier counts, one completing earlier does not.
        assert_eq!(run(&stop, &[0, 1, 2], 4), None);
        assert_eq!(
            run(&stop, &[0, 1, 2], 3),
            Some((2, Stopped::String { index: 0, cut: 3 }))
        );
        // Stop tokens: not emitted, any position.
        let stop = stop.with_tokens([5]);
        assert_eq!(run(&stop, &[0, 5], 0), Some((1, Stopped::Token(5))));
        // An id outside the table has no bytes.
        assert_eq!(run(&stop, &[0, 99, 1, 2], 0).unwrap().0, 3);
    }

    #[test]
    fn the_earliest_start_wins_then_the_lowest_index() {
        let vocab: &[&[u8]] = &[b"xyz", b"abcd"];
        // "bcd" starts before "cd"; both end in token 1.
        let stop = stop_over(vocab, &["cd", "bcd", "zab"], 0);
        assert_eq!(
            run(&stop, &[0, 1], 0),
            Some((1, Stopped::String { index: 2, cut: 5 }))
        );
        let stop = stop_over(vocab, &["cd", "bcd"], 0);
        assert_eq!(
            run(&stop, &[0, 1], 0),
            Some((1, Stopped::String { index: 1, cut: 3 }))
        );
        // Equal strings: the lower index.
        let stop = stop_over(vocab, &["cd", "cd"], 0);
        assert_eq!(
            run(&stop, &[1], 0),
            Some((0, Stopped::String { index: 0, cut: 2 }))
        );
    }

    #[test]
    fn leading_spaces_are_stripped_as_the_decoder_strips_them() {
        // SentencePiece-style: one leading space of the text is not part of it.
        let vocab: &[&[u8]] = &[b" ", b" Hi", b"!"];
        let stop = stop_over(vocab, &[" Hi"], 1);
        // The first token " " is the stripped space, so the second keeps its own: " Hi".
        assert_eq!(
            run(&stop, &[0, 1], 0),
            Some((1, Stopped::String { index: 0, cut: 3 }))
        );
        // As the first token, " Hi" decodes as "Hi": no match.
        assert_eq!(run(&stop, &[1, 2], 0), None);
    }

    #[test]
    fn matching_agrees_with_a_whole_text_search() {
        // Random token streams over a small alphabet, against the reference.
        let vocab: Vec<Vec<u8>> = ["a", "b", "ab", "ba", "aab", "\n", "a\n", "", "bb"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let refs: Vec<&[u8]> = vocab.iter().map(Vec::as_slice).collect();
        let strings = ["aba", "b\na", "bbb", "\n\n"];
        let stop = stop_over(&refs, &strings, 0);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2000 {
            let tokens: Vec<u32> = (0..12)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % vocab.len() as u64) as u32
                })
                .collect();
            let text: Vec<u8> = tokens
                .iter()
                .flat_map(|&t| vocab[t as usize].clone())
                .collect();
            let got = run(&stop, &tokens, 0);
            match reference(&text, &strings) {
                None => assert_eq!(got, None, "{tokens:?}"),
                Some((index, start)) => {
                    let (at, m) = got.unwrap_or_else(|| panic!("{tokens:?}"));
                    let end: usize = tokens[..=at].iter().map(|&t| vocab[t as usize].len()).sum();
                    assert_eq!(
                        m,
                        Stopped::String {
                            index,
                            cut: end - start
                        },
                        "{tokens:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn stop_strings_are_validated() {
        let tokenizer = crate::tokenizer::tests::tiny();
        let e = Stop::strings(&tokenizer, [""]).unwrap_err().to_string();
        assert!(e.contains("string 0 is empty"), "{e}");
        let long = "x".repeat(MAX_STOP_BYTES + 1);
        let e = Stop::strings(&tokenizer, ["ok", &long])
            .unwrap_err()
            .to_string();
        assert!(e.contains("string 1 is longer"), "{e}");
        let many = vec!["s"; MAX_STOP_STRINGS + 1];
        assert!(Stop::strings(&tokenizer, many).is_err());
        let a = Stop::strings(&tokenizer, ["\n\n"]).unwrap();
        assert_eq!(a.clone(), a);
        assert_ne!(a, Stop::strings(&tokenizer, ["\n\n"]).unwrap());
        assert!(Stop::default().is_empty() && !a.is_empty());
    }

    #[test]
    fn text_streams_never_show_a_stop_string() {
        let tokenizer = crate::tokenizer::tests::tiny();
        let text = "Hello world.\n\nUser: héllo again";
        let ids = tokenizer.encode_text(text);
        for (stop, want) in [
            ("\n\nUser:", "Hello world."),
            ("é", "Hello world.\n\nUser: h"),
            ("lo w", "Hel"),
            ("absent", text),
        ] {
            let options = GenerateOptions {
                stop: Stop::strings(&tokenizer, [stop]).unwrap(),
                ..Default::default()
            };
            let mut stream = TextStream::new(&tokenizer, &options);
            let mut shown = String::new();
            for &id in &ids {
                shown.push_str(&stream.push(id));
                // Nothing shown so far contains any part that the stop string begins with
                // at its end, beyond what the final text keeps.
                assert!(want.starts_with(&shown), "{stop:?}: {shown:?}");
            }
            shown.push_str(&stream.finish());
            assert_eq!(shown, want, "{stop:?}");
            // The cut agrees: decoded output up to the completing token, minus `cut`.
            let mut s = StopState::default();
            let hit = ids
                .iter()
                .enumerate()
                .find_map(|(i, &t)| s.push(&options.stop, t, i + 1, 0).map(|m| (i, m)));
            if let Some((i, Stopped::String { cut, .. })) = hit {
                let bytes = tokenizer.decode_bytes(&ids[..=i]);
                assert_eq!(&bytes[..bytes.len() - cut], want.as_bytes(), "{stop:?}");
            } else {
                assert_eq!(want, text);
            }
        }
    }
}
