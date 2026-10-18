//! Byte-level BPE tokenizers read from Hugging Face `tokenizer.json` files (the family of
//! GPT-2, Llama 3, Qwen 2/3 and Mistral's Tekken).
//!
//! Encoding follows the `tokenizers` library: added tokens are split off first; the rest
//! is normalized (NFC, when the file says so), split by the pre-tokenizer's pattern (the
//! GPT-2 and GPT-4-style patterns, matched by a hand-written scanner with the same
//! leftmost-first semantics as the regular expression), mapped to byte-level characters and
//! merged by rank.

use std::collections::HashMap;
use std::path::Path;

use nuvec::io::json::{self, Json};
use nuvec::{Error, Result};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("tokenizer.json: {msg}"))
}

/// The pre-tokenizer's split pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
    Gpt2,
    /// GPT-4 style (Llama 3, Qwen): case-insensitive contractions, letters with one
    /// leading non-letter, runs of at most `digits` digits, punctuation with trailing
    /// newlines, newlines, then whitespace.
    Gpt4 {
        /// The longest digit run (`\p{N}` is 1, `\p{N}{1,3}` is 3).
        digits: usize,
    },
}

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const GPT4_PREFIX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}";
const GPT4_SUFFIX: &str = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Pattern {
    fn parse(regex: &str) -> Result<Self> {
        if regex == GPT2 {
            return Ok(Self::Gpt2);
        }
        let digits = regex
            .strip_prefix(GPT4_PREFIX)
            .and_then(|rest| rest.strip_suffix(GPT4_SUFFIX))
            .and_then(|quantifier| match quantifier {
                "" => Some(1),
                "{1,3}" => Some(3),
                _ => None,
            });
        digits
            .map(|digits| Self::Gpt4 { digits })
            .ok_or_else(|| invalid(format!("unsupported pre-tokenizer pattern {regex:?}")))
    }

    /// The pieces of `text`, in order (they cover it exactly).
    fn split(self, text: &str) -> Vec<&str> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let n = match self {
                Self::Gpt2 => gpt2_match(&chars, i),
                Self::Gpt4 { digits } => gpt4_match(&chars, i, digits),
            }
            .max(1);
            let start = chars[i].0;
            let end = chars.get(i + n).map_or(text.len(), |c| c.0);
            pieces.push(&text[start..end]);
            i += n;
        }
        pieces
    }
}

fn is_letter(c: char) -> bool {
    use GeneralCategory as G;
    matches!(
        get_general_category(c),
        G::UppercaseLetter
            | G::LowercaseLetter
            | G::TitlecaseLetter
            | G::ModifierLetter
            | G::OtherLetter
    )
}

/// `\p{N}`: decimal digits, letter numbers and other numbers.
fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

fn run(chars: &[(usize, char)], from: usize, f: impl Fn(char) -> bool) -> usize {
    chars[from..].iter().take_while(|(_, c)| f(*c)).count()
}

/// A contraction at `i` (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), case-insensitively
/// or not: its length.
fn contraction(chars: &[(usize, char)], i: usize, any_case: bool) -> usize {
    if chars[i].1 != '\'' {
        return 0;
    }
    let at = |k: usize| {
        chars
            .get(i + k)
            .map(|&(_, c)| if any_case { c.to_ascii_lowercase() } else { c })
    };
    for word in ["s", "t", "re", "ve", "m", "ll", "d"] {
        if word.chars().enumerate().all(|(k, w)| at(k + 1) == Some(w)) {
            return 1 + word.len();
        }
    }
    0
}

/// `\s+(?!\S)` then `\s+`: whitespace at `i`, leaving the last space before a
/// non-space to the next piece.
fn spaces(chars: &[(usize, char)], i: usize) -> usize {
    let n = run(chars, i, char::is_whitespace);
    if n == 0 {
        return 0;
    }
    if i + n == chars.len() || n == 1 {
        n
    } else {
        n - 1
    }
}

fn gpt2_match(chars: &[(usize, char)], i: usize) -> usize {
    let c = chars[i].1;
    let n = contraction(chars, i, false);
    if n > 0 {
        return n;
    }
    let space = usize::from(c == ' ');
    for class in [is_letter as fn(char) -> bool, is_number, is_other] {
        let n = run(chars, i + space, class);
        if n > 0 {
            return space + n;
        }
    }
    spaces(chars, i)
}

fn gpt4_match(chars: &[(usize, char)], i: usize, digits: usize) -> usize {
    let c = chars[i].1;
    let n = contraction(chars, i, true);
    if n > 0 {
        return n;
    }
    // `[^\r\n\p{L}\p{N}]?\p{L}+`
    if !matches!(c, '\r' | '\n') && !is_letter(c) && !is_number(c) {
        let n = run(chars, i + 1, is_letter);
        if n > 0 {
            return 1 + n;
        }
    }
    let n = run(chars, i, is_letter);
    if n > 0 {
        return n;
    }
    // `\p{N}{1,digits}`
    let n = run(chars, i, is_number);
    if n > 0 {
        return n.min(digits);
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let space = usize::from(c == ' ');
    let n = run(chars, i + space, is_other);
    if n > 0 {
        let end = i + space + n;
        return space + n + run(chars, end, |c| matches!(c, '\r' | '\n'));
    }
    // `\s*[\r\n]+`: through the last newline of the whitespace run.
    let ws = run(chars, i, char::is_whitespace);
    if let Some(last) = (i..i + ws)
        .rev()
        .find(|&k| matches!(chars[k].1, '\r' | '\n'))
    {
        return last + 1 - i;
    }
    spaces(chars, i)
}

/// GPT-2's map from bytes to printable characters (byte-level BPE).
fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let printable = matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        table[b as usize] = if printable {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(next - 1).expect("valid")
        };
    }
    table
}

/// A byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// Token strings by id (byte-level characters, or added tokens' text).
    tokens: Vec<Option<String>>,
    /// `(left, right) -> (rank, merged)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens (special or not), longest first, with whether they are special.
    added: Vec<(String, u32, bool)>,
    nfc: bool,
    pattern: Pattern,
    ignore_merges: bool,
    to_char: [char; 256],
    to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Reads a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", path.display())))?;
        Self::from_json(&json::parse(&bytes)?)
    }

    /// Builds a tokenizer from a parsed `tokenizer.json`.
    pub fn from_json(file: &Json) -> Result<Self> {
        let model = file.get("model").ok_or_else(|| invalid("no model"))?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return Err(invalid("only BPE models are supported"));
        }
        if model.get("byte_fallback").and_then(Json::as_bool) == Some(true) {
            return Err(invalid(
                "byte-fallback (SentencePiece-style) BPE is not supported",
            ));
        }
        let mut vocab = HashMap::new();
        let mut tokens: Vec<Option<String>> = Vec::new();
        let put = |tokens: &mut Vec<Option<String>>, s: &str, id: u32| {
            if tokens.len() <= id as usize {
                tokens.resize(id as usize + 1, None);
            }
            tokens[id as usize] = Some(s.to_owned());
        };
        for (token, id) in model
            .get("vocab")
            .and_then(Json::as_object)
            .ok_or_else(|| invalid("no vocab"))?
        {
            let id = id.as_i64().ok_or_else(|| invalid("bad id"))? as u32;
            vocab.insert(token.clone(), id);
            put(&mut tokens, token, id);
        }
        let mut merges = HashMap::new();
        for (rank, m) in model
            .get("merges")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let (a, b) = match m {
                Json::Str(s) => s.split_once(' ').ok_or_else(|| invalid("bad merge"))?,
                Json::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or_else(|| invalid("bad merge"))?,
                    pair[1].as_str().ok_or_else(|| invalid("bad merge"))?,
                ),
                _ => return Err(invalid("bad merge")),
            };
            let id = |s: &str| vocab.get(s).copied();
            if let (Some(l), Some(r), Some(m)) = (id(a), id(b), id(&format!("{a}{b}"))) {
                merges.insert((l, r), (rank as u32, m));
            }
        }
        let mut added = Vec::new();
        for t in file
            .get("added_tokens")
            .and_then(Json::as_array)
            .unwrap_or_default()
        {
            let content = t.get("content").and_then(Json::as_str);
            let id = t.get("id").and_then(Json::as_i64);
            if let (Some(content), Some(id)) = (content, id) {
                let special = t.get("special").and_then(Json::as_bool).unwrap_or(false);
                added.push((content.to_owned(), id as u32, special));
                put(&mut tokens, content, id as u32);
            }
        }
        added.sort_by_key(|(c, _, _)| std::cmp::Reverse(c.len()));
        let nfc = match file.get("normalizer") {
            None | Some(Json::Null) => false,
            Some(n) => match n.get("type").and_then(Json::as_str) {
                Some("NFC") => true,
                _ => return Err(invalid("unsupported normalizer")),
            },
        };
        let pattern = Self::pattern(file.get("pre_tokenizer"))?;
        let to_char = byte_chars();
        let to_byte = to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        Ok(Self {
            vocab,
            tokens,
            merges,
            added,
            nfc,
            pattern,
            ignore_merges: model
                .get("ignore_merges")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            to_char,
            to_byte,
        })
    }

    /// The split pattern of a `Split` + `ByteLevel` sequence, or of a `ByteLevel`
    /// pre-tokenizer with its own (GPT-2) pattern.
    fn pattern(pre: Option<&Json>) -> Result<Pattern> {
        let pre = pre.ok_or_else(|| invalid("no pre_tokenizer"))?;
        let kind = pre.get("type").and_then(Json::as_str);
        match kind {
            Some("ByteLevel") if pre.get("use_regex").and_then(Json::as_bool) != Some(false) => {
                if pre.get("add_prefix_space").and_then(Json::as_bool) == Some(true) {
                    return Err(invalid("add_prefix_space is not supported"));
                }
                Ok(Pattern::Gpt2)
            }
            Some("Sequence") => {
                let parts = pre
                    .get("pretokenizers")
                    .and_then(Json::as_array)
                    .unwrap_or_default();
                let mut pattern = None;
                for p in parts {
                    match p.get("type").and_then(Json::as_str) {
                        Some("Split") => {
                            // The supported patterns match every character, so isolating
                            // the matches equals removing the inverted gaps.
                            let behavior = p.get("behavior").and_then(Json::as_str);
                            let invert = p.get("invert").and_then(Json::as_bool).unwrap_or(false);
                            match (behavior, invert) {
                                (Some("Isolated") | None, false) | (Some("Removed"), true) => {}
                                other => {
                                    return Err(invalid(format!("unsupported split {other:?}")));
                                }
                            }
                            let regex = p
                                .get("pattern")
                                .and_then(|r| r.get("Regex"))
                                .and_then(Json::as_str)
                                .ok_or_else(|| invalid("split without a regex"))?;
                            pattern = Some(Pattern::parse(regex)?);
                        }
                        Some("ByteLevel")
                            if p.get("use_regex").and_then(Json::as_bool) == Some(false) => {}
                        other => {
                            return Err(invalid(format!("unsupported pre-tokenizer {other:?}")));
                        }
                    }
                }
                pattern.ok_or_else(|| invalid("no split pattern"))
            }
            other => Err(invalid(format!("unsupported pre-tokenizer {other:?}"))),
        }
    }

    /// The id of a token string (an added token's text or a byte-level vocabulary entry).
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(c, _, _)| c == token)
            .map(|(_, id, _)| *id)
            .or_else(|| self.vocab.get(token).copied())
    }

    /// The number of ids (the largest id plus one).
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The ids of `text`. Added tokens (`<|im_start|>`, …) in the text are encoded as such.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest added token (the longest at a position).
            let next = self
                .added
                .iter()
                .filter_map(|(c, id, _)| rest.find(c.as_str()).map(|at| (at, c.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let (plain, after) = match next {
                Some((at, len, id)) => {
                    let plain = &rest[..at];
                    self.encode_plain(plain, &mut ids);
                    ids.push(id);
                    (None, &rest[at + len..])
                }
                None => (Some(rest), ""),
            };
            if let Some(plain) = plain {
                self.encode_plain(plain, &mut ids);
            }
            rest = after;
        }
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let normalized: String;
        let text = if self.nfc {
            normalized = text.nfc().collect();
            &normalized
        } else {
            text
        };
        for piece in self.pattern.split(text) {
            let word: String = piece.bytes().map(|b| self.to_char[b as usize]).collect();
            if self.ignore_merges
                && let Some(&id) = self.vocab.get(&word)
            {
                ids.push(id);
                continue;
            }
            ids.extend(self.bpe(&word));
        }
    }

    /// Merges a word's byte-level characters by rank (lowest first, leftmost on ties).
    fn bpe(&self, word: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = word
            .chars()
            .filter_map(|c| self.vocab.get(c.encode_utf8(&mut [0; 4]) as &str).copied())
            .collect();
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(r, m)| (r, i, m)))
                .min();
            let Some((_, i, merged)) = best else {
                return symbols;
            };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
    }

    /// The bytes of `ids` (added tokens as their text).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(Some(token)) = self.tokens.get(id as usize) else {
                continue;
            };
            if self.added.iter().any(|(_, a, _)| *a == id) {
                out.extend_from_slice(token.as_bytes());
            } else {
                out.extend(token.chars().filter_map(|c| self.to_byte.get(&c)));
            }
        }
        out
    }

    /// The text of `ids` (invalid UTF-8, such as a split character, is replaced).
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// Turns a stream of ids into text, holding back bytes until they form whole characters.
#[derive(Debug, Default, Clone)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// The text completed by `id`.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> String {
        self.pending.extend(tokenizer.decode_bytes(&[id]));
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // An invalid sequence (not just an unfinished one): emit it replaced.
            Err(_) => self.pending.len(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        text
    }

    /// The text of any bytes still held back.
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(p: Pattern, s: &str) -> Vec<&str> {
        p.split(s)
    }

    #[test]
    fn gpt4_pattern_splits_like_the_regex() {
        let qwen = Pattern::Gpt4 { digits: 1 };
        assert_eq!(
            pieces(qwen, "Hello world, it's 2024!\n\n  ok  "),
            [
                "Hello", " world", ",", " it", "'s", " ", "2", "0", "2", "4", "!\n\n", " ", " ok",
                "  "
            ]
        );
        let llama3 = Pattern::Gpt4 { digits: 3 };
        assert_eq!(pieces(llama3, "12345 x"), ["123", "45", " x"]);
        // Case-insensitive contractions; a leading non-letter joins a word.
        assert_eq!(pieces(qwen, "I'LL (go)"), ["I", "'LL", " (", "go", ")"]);
        assert_eq!(pieces(qwen, "a\n\n b"), ["a", "\n\n", " b"]);
        assert_eq!(pieces(qwen, "x \n y"), ["x", " \n", " y"]);
        assert_eq!(pieces(qwen, "é漢字"), ["é漢字"]);
    }

    #[test]
    fn gpt2_pattern_splits_like_the_regex() {
        assert_eq!(
            pieces(Pattern::Gpt2, "Hello  world's 123!?"),
            ["Hello", " ", " world", "'s", " 123", "!?"]
        );
    }

    #[test]
    fn patterns_are_recognised() {
        let qwen = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(Pattern::parse(qwen).unwrap(), Pattern::Gpt4 { digits: 1 });
        assert!(Pattern::parse(r"\w+").is_err());
    }

    /// A tiny byte-level tokenizer: the 256 byte characters plus merges for "he", "ll",
    /// "hell" and "hello", and one added token.
    fn tiny() -> Tokenizer {
        let chars = byte_chars();
        let mut vocab: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        vocab.extend(["he", "ll", "hell", "hello"].map(String::from));
        let entries: Vec<String> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}:{i}", json_quote(t)))
            .collect();
        let text = format!(
            r#"{{"normalizer":{{"type":"NFC"}},
              "pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}},
              "added_tokens":[{{"id":300,"content":"<|end|>","special":true}}],
              "model":{{"type":"BPE","vocab":{{{}}},"merges":["h e","l l","he ll",["hell","o"]]}}}}"#,
            entries.join(",")
        );
        Tokenizer::from_json(&json::parse(text.as_bytes()).unwrap()).unwrap()
    }

    fn json_quote(s: &str) -> String {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    #[test]
    fn encodes_and_decodes_byte_level_bpe() {
        let t = tiny();
        let ids = t.encode("hello<|end|>hell é");
        let hello = t.token_id("hello").unwrap();
        let hell = t.token_id("hell").unwrap();
        assert_eq!(ids[..3], [hello, 300, hell]);
        assert_eq!(t.decode(&ids), "hello<|end|>hell é");
        // Streaming holds back a split two-byte character.
        let mut s = StreamDecoder::default();
        let text: String = ids.iter().map(|&id| s.push(&t, id)).collect();
        assert_eq!(text + &s.finish(), "hello<|end|>hell é");
        let e_bytes = t.encode("é");
        assert_eq!(e_bytes.len(), 2);
        let mut s = StreamDecoder::default();
        assert_eq!(s.push(&t, e_bytes[0]), "");
        assert_eq!(s.push(&t, e_bytes[1]), "é");
    }
}
