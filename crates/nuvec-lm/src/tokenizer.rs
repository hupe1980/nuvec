//! Byte-level BPE tokenizers read from Hugging Face `tokenizer.json` files: those whose
//! pre-tokenizer splits by the GPT-2 pattern or a GPT-4-style one (GPT-2, Llama 3, Qwen
//! 2/3/3.5). Other split patterns (Mistral's Tekken among them) and SentencePiece-style
//! byte-fallback BPE (Llama 2, Mistral v0.1–v0.3) are rejected when the file is read.
//!
//! Encoding follows the `tokenizers` library: added tokens are split off first (in one
//! pass, the longest match at the earliest position); the rest is normalized (NFC, when
//! the file says so), split by the pre-tokenizer's pattern (matched by a hand-written
//! scanner with the same leftmost-first semantics as the regular expression), mapped to
//! byte-level characters and merged by rank (lowest first, with a priority queue, and a
//! cache of recent words), so encoding time grows about linearly with the text.

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Mutex, PoisonError};

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
        /// Words also take combining marks, which are not punctuation (`[\p{L}\p{M}]+`,
        /// Qwen 3.5).
        marks: bool,
    },
}

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const GPT4_PREFIX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}";
const GPT4_SUFFIX: &str = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const MARKS_PREFIX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}";
const MARKS_SUFFIX: &str = r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Pattern {
    fn parse(regex: &str) -> Result<Self> {
        if regex == GPT2 {
            return Ok(Self::Gpt2);
        }
        let quantified = |prefix: &str, suffix: &str| {
            regex
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_suffix(suffix))
                .and_then(|quantifier| match quantifier {
                    "" => Some(1),
                    "{1,3}" => Some(3),
                    _ => None,
                })
        };
        let gpt4 = quantified(GPT4_PREFIX, GPT4_SUFFIX).map(|digits| (digits, false));
        let marks = quantified(MARKS_PREFIX, MARKS_SUFFIX).map(|digits| (digits, true));
        gpt4.or(marks)
            .map(|(digits, marks)| Self::Gpt4 { digits, marks })
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
                Self::Gpt4 { digits, marks } => gpt4_match(&chars, i, digits, marks),
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

/// `\p{M}`: combining marks.
fn is_mark(c: char) -> bool {
    use GeneralCategory as G;
    matches!(
        get_general_category(c),
        G::NonspacingMark | G::SpacingMark | G::EnclosingMark
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

fn gpt4_match(chars: &[(usize, char)], i: usize, digits: usize, marks: bool) -> usize {
    let c = chars[i].1;
    let n = contraction(chars, i, true);
    if n > 0 {
        return n;
    }
    // `\p{L}`, or `[\p{L}\p{M}]` with marks.
    let word = |c: char| is_letter(c) || (marks && is_mark(c));
    // `[^\r\n\p{L}\p{N}]?\p{L}+`
    if !matches!(c, '\r' | '\n') && !is_letter(c) && !is_number(c) {
        let n = run(chars, i + 1, word);
        if n > 0 {
            return 1 + n;
        }
    }
    let n = run(chars, i, word);
    if n > 0 {
        return n;
    }
    // `\p{N}{1,digits}`
    let n = run(chars, i, is_number);
    if n > 0 {
        return n.min(digits);
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*` (and not `\p{M}`, with marks)
    let space = usize::from(c == ' ');
    let n = run(chars, i + space, |c| is_other(c) && !(marks && is_mark(c)));
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
    /// The added tokens (indices into `added`, longest first) by their first byte.
    added_by_byte: HashMap<u8, Vec<usize>>,
    /// The added tokens' ids.
    added_ids: HashSet<u32>,
    nfc: bool,
    pattern: Pattern,
    ignore_merges: bool,
    to_char: [char; 256],
    to_byte: HashMap<char, u8>,
    /// Special tokens the post-processor puts around a single sequence (Llama's
    /// `<|begin_of_text|>`): before and after it.
    template: Template,
    /// The ids of recently encoded words.
    words: WordCache,
}

/// Words cached at most (as the `tokenizers` library's BPE cache).
const WORD_CACHE_CAPACITY: usize = 10_000;
/// Longer words (in bytes) are not cached.
const WORD_CACHE_LONGEST: usize = 256;

/// The ids of the most recently used words (least recently used evicted first), shared
/// by the threads encoding with one tokenizer. A clone starts empty.
#[derive(Default)]
struct WordCache(Mutex<Lru>);

#[derive(Default)]
struct Lru {
    /// Each word's ids and when it was last used.
    words: HashMap<String, (Vec<u32>, u64)>,
    /// The words by when they were last used.
    order: BTreeMap<u64, String>,
    clock: u64,
}

impl Lru {
    fn get(&mut self, word: &str) -> Option<Vec<u32>> {
        let (ids, used) = self.words.get_mut(word)?;
        let key = self.order.remove(used).expect("ordered");
        self.clock += 1;
        *used = self.clock;
        self.order.insert(self.clock, key);
        Some(ids.clone())
    }

    fn insert(&mut self, word: &str, ids: &[u32]) {
        if word.len() > WORD_CACHE_LONGEST || self.words.contains_key(word) {
            return;
        }
        if self.words.len() >= WORD_CACHE_CAPACITY
            && let Some((_, oldest)) = self.order.pop_first()
        {
            self.words.remove(&oldest);
        }
        self.clock += 1;
        self.words
            .insert(word.to_owned(), (ids.to_vec(), self.clock));
        self.order.insert(self.clock, word.to_owned());
    }
}

impl Clone for WordCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for WordCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let words = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .words
            .len();
        f.debug_struct("WordCache").field("words", &words).finish()
    }
}

/// Special tokens (text and id) before and after a sequence.
type Template = (Vec<(String, u32)>, Vec<(String, u32)>);

/// The special tokens around sequence `A` in a `TemplateProcessing` post-processor (also
/// inside a `Sequence`).
fn template(post: Option<&Json>) -> Template {
    let Some(post) = post else {
        return Default::default();
    };
    if post.get("type").and_then(Json::as_str) == Some("Sequence") {
        return post
            .get("processors")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .iter()
            .map(|p| template(Some(p)))
            .find(|(before, after)| !before.is_empty() || !after.is_empty())
            .unwrap_or_default();
    }
    if post.get("type").and_then(Json::as_str) != Some("TemplateProcessing") {
        return Default::default();
    }
    let special = post.get("special_tokens");
    let (mut before, mut after, mut seen) = (Vec::new(), Vec::new(), false);
    for piece in post
        .get("single")
        .and_then(Json::as_array)
        .unwrap_or_default()
    {
        if piece.get("Sequence").is_some() {
            seen = true;
        } else if let Some(name) = piece
            .get("SpecialToken")
            .and_then(|t| t.get("id"))
            .and_then(Json::as_str)
        {
            let ids = special
                .and_then(|s| s.get(name))
                .and_then(|t| t.get("ids"))
                .and_then(Json::as_array)
                .unwrap_or_default();
            for id in ids.iter().filter_map(|id| token_id(id.as_i64()?).ok()) {
                let side = if seen { &mut after } else { &mut before };
                side.push((name.to_owned(), id));
            }
        }
    }
    (before, after)
}

/// The largest token id a `tokenizer.json` may use. Real vocabularies stay below 2^20;
/// the cap keeps a hostile file from sizing the id table to gigabytes.
const MAX_TOKEN_ID: i64 = (1 << 24) - 1;

/// A token id from `tokenizer.json`, checked to lie in `0..=MAX_TOKEN_ID`.
fn token_id(id: i64) -> Result<u32> {
    if (0..=MAX_TOKEN_ID).contains(&id) {
        Ok(id as u32)
    } else {
        Err(invalid(format!("token id {id} out of range")))
    }
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
            let id = token_id(id.as_i64().ok_or_else(|| invalid("bad id"))?)?;
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
            if let (Some(content), Some(id)) = (content, id)
                && !content.is_empty()
            {
                let special = t.get("special").and_then(Json::as_bool).unwrap_or(false);
                let id = token_id(id)?;
                added.push((content.to_owned(), id, special));
                put(&mut tokens, content, id);
            }
        }
        added.sort_by_key(|(c, _, _)| std::cmp::Reverse(c.len()));
        let mut added_by_byte: HashMap<u8, Vec<usize>> = HashMap::new();
        for (i, (content, _, _)) in added.iter().enumerate() {
            added_by_byte
                .entry(content.as_bytes()[0])
                .or_default()
                .push(i);
        }
        let added_ids = added.iter().map(|&(_, id, _)| id).collect();
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
            added_by_byte,
            added_ids,
            nfc,
            pattern,
            ignore_merges: model
                .get("ignore_merges")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            to_char,
            to_byte,
            template: template(file.get("post_processor")),
            words: WordCache::default(),
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

    /// The ids of `text`, with the special tokens the tokenizer's post-processor adds
    /// (Llama's `<|begin_of_text|>`) unless the text already starts with them, as mlx-lm
    /// encodes prompts. Added tokens (`<|im_start|>`, …) in the text are encoded as such.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let (before, after) = &self.template;
        let present = !before.is_empty() && text.starts_with(&before[0].0);
        if before.is_empty() && after.is_empty() || present {
            return self.encode_text(text);
        }
        let mut ids: Vec<u32> = before.iter().map(|(_, id)| *id).collect();
        ids.extend(self.encode_text(text));
        ids.extend(after.iter().map(|(_, id)| *id));
        ids
    }

    /// The ids of `text` alone, without the post-processor's special tokens.
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let bytes = text.as_bytes();
        // One pass: at each position, the longest added token starting there (the earliest
        // match wins). A match starts at a character boundary: its first byte begins a
        // character.
        let (mut plain, mut i) = (0, 0);
        while i < bytes.len() {
            let found = self.added_by_byte.get(&bytes[i]).and_then(|candidates| {
                candidates
                    .iter()
                    .map(|&k| &self.added[k])
                    .find(|(c, _, _)| bytes[i..].starts_with(c.as_bytes()))
            });
            match found {
                Some((content, id, _)) => {
                    self.encode_plain(&text[plain..i], &mut ids);
                    ids.push(*id);
                    i += content.len();
                    plain = i;
                }
                None => i += 1,
            }
        }
        self.encode_plain(&text[plain..], &mut ids);
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
        let mut words = self.words.0.lock().unwrap_or_else(PoisonError::into_inner);
        for piece in self.pattern.split(text) {
            let word: String = piece.bytes().map(|b| self.to_char[b as usize]).collect();
            if self.ignore_merges
                && let Some(&id) = self.vocab.get(&word)
            {
                ids.push(id);
                continue;
            }
            if let Some(cached) = words.get(&word) {
                ids.extend(cached);
                continue;
            }
            let merged = self.bpe(&word);
            words.insert(&word, &merged);
            ids.extend(merged);
        }
    }

    /// Merges a word's byte-level characters by rank (lowest first, leftmost on ties):
    /// the candidate merges in a priority queue over a linked list of symbols, so a word
    /// of n characters takes O(n log n) (a merge that no longer applies is skipped when
    /// it comes up).
    fn bpe(&self, word: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = word
            .chars()
            .filter_map(|c| self.vocab.get(c.encode_utf8(&mut [0; 4]) as &str).copied())
            .collect();
        let n = symbols.len();
        if n < 2 {
            return symbols;
        }
        const NONE: usize = usize::MAX;
        // Neighbours in the list (`NONE` at the ends); merged-away symbols are dead.
        let mut prev: Vec<usize> = (0..n).map(|i| i.checked_sub(1).unwrap_or(NONE)).collect();
        let mut next: Vec<usize> = (1..=n).map(|i| if i < n { i } else { NONE }).collect();
        let mut alive = vec![true; n];
        let mut queue = BinaryHeap::new();
        let rank = |symbols: &[u32], i: usize, j: usize| {
            self.merges.get(&(symbols[i], symbols[j])).map(|&(r, _)| r)
        };
        for i in 0..n - 1 {
            if let Some(r) = rank(&symbols, i, i + 1) {
                queue.push(std::cmp::Reverse((r, i)));
            }
        }
        while let Some(std::cmp::Reverse((r, i))) = queue.pop() {
            let j = next[i];
            // Stale: `i` was merged away, or its pair changed since.
            if !alive[i] || j == NONE || rank(&symbols, i, j) != Some(r) {
                continue;
            }
            symbols[i] = self.merges[&(symbols[i], symbols[j])].1;
            alive[j] = false;
            next[i] = next[j];
            if next[i] != NONE {
                prev[next[i]] = i;
            }
            if prev[i] != NONE
                && let Some(r) = rank(&symbols, prev[i], i)
            {
                queue.push(std::cmp::Reverse((r, prev[i])));
            }
            if next[i] != NONE
                && let Some(r) = rank(&symbols, i, next[i])
            {
                queue.push(std::cmp::Reverse((r, i)));
            }
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i != NONE {
            out.push(symbols[i]);
            i = next[i];
        }
        out
    }

    /// The bytes of `ids` (added tokens as their text).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(Some(token)) = self.tokens.get(id as usize) else {
                continue;
            };
            if self.added_ids.contains(&id) {
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
    /// The text completed by `id`: whole characters, an invalid byte sequence as one
    /// U+FFFD, and an unfinished character at the end held back.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> String {
        self.pending.extend(tokenizer.decode_bytes(&[id]));
        let mut text = String::new();
        let mut done = 0;
        loop {
            match std::str::from_utf8(&self.pending[done..]) {
                Ok(s) => {
                    text.push_str(s);
                    done = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = done + e.valid_up_to();
                    text.push_str(
                        std::str::from_utf8(&self.pending[done..valid]).expect("valid up to"),
                    );
                    done = valid;
                    match e.error_len() {
                        // An invalid sequence: replaced, then the rest.
                        Some(n) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            done += n;
                        }
                        // An unfinished character: held back.
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..done);
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
        let qwen = Pattern::Gpt4 {
            digits: 1,
            marks: false,
        };
        assert_eq!(
            pieces(qwen, "Hello world, it's 2024!\n\n  ok  "),
            [
                "Hello", " world", ",", " it", "'s", " ", "2", "0", "2", "4", "!\n\n", " ", " ok",
                "  "
            ]
        );
        let llama3 = Pattern::Gpt4 {
            digits: 3,
            marks: false,
        };
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

    /// Qwen 3.5's pattern, against Hugging Face `tokenizers` (`Split`, isolated).
    #[test]
    fn marks_pattern_splits_like_the_regex() {
        let qwen35 = Pattern::Gpt4 {
            digits: 1,
            marks: true,
        };
        let cases: [(&str, &[&str]); 5] = [
            (
                "cafe\u{301} ete\u{301}!",
                &["cafe\u{301}", " ete\u{301}", "!"],
            ),
            ("नमस्ते दुनिया", &["नमस्ते", " दुनिया"]),
            ("x\u{301}?\u{301} a", &["x\u{301}", "?\u{301}", " a"]),
            (
                "\u{301}\u{301}b 12.5%",
                &["\u{301}\u{301}b", " ", "1", "2", ".", "5", "%"],
            ),
            (
                "Hello world, it's 2024!\n\n  ok  ",
                &[
                    "Hello", " world", ",", " it", "'s", " ", "2", "0", "2", "4", "!\n\n", " ",
                    " ok", "  ",
                ],
            ),
        ];
        for (text, want) in cases {
            assert_eq!(pieces(qwen35, text), want, "{text:?}");
        }
    }

    #[test]
    fn patterns_are_recognised() {
        let qwen = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(
            Pattern::parse(qwen).unwrap(),
            Pattern::Gpt4 {
                digits: 1,
                marks: false
            }
        );
        let qwen35 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(
            Pattern::parse(qwen35).unwrap(),
            Pattern::Gpt4 {
                digits: 1,
                marks: true
            }
        );
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

    /// Llama's post-processor puts `<|begin_of_text|>` first, unless the text has it.
    #[test]
    fn template_special_tokens_are_added_once() {
        let chars = byte_chars();
        let entries: Vec<String> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}:{i}", json_quote(&c.to_string())))
            .collect();
        let text = format!(
            r#"{{"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}},
              "added_tokens":[{{"id":300,"content":"<|bos|>","special":true}}],
              "post_processor":{{"type":"Sequence","processors":[{{"type":"ByteLevel"}},
                {{"type":"TemplateProcessing","single":[{{"SpecialToken":{{"id":"<|bos|>","type_id":0}}}},
                  {{"Sequence":{{"id":"A","type_id":0}}}}],
                  "special_tokens":{{"<|bos|>":{{"id":"<|bos|>","ids":[300],"tokens":["<|bos|>"]}}}}}}]}},
              "model":{{"type":"BPE","vocab":{{{}}},"merges":[]}}}}"#,
            entries.join(",")
        );
        let t = Tokenizer::from_json(&json::parse(text.as_bytes()).unwrap()).unwrap();
        assert_eq!(
            t.encode("hi"),
            [300, t.token_id("h").unwrap(), t.token_id("i").unwrap()]
        );
        assert_eq!(t.encode("<|bos|>hi"), t.encode("hi"));
        assert_eq!(t.encode_text("hi").len(), 2);
        assert_eq!(tiny().encode("hello"), tiny().encode_text("hello"));
    }

    #[test]
    fn out_of_range_token_ids_are_errors() {
        for id in ["-1", "4294967296", "16777216"] {
            for text in [
                format!(r#"{{"model":{{"type":"BPE","vocab":{{"a":{id}}},"merges":[]}}}}"#),
                format!(
                    r#"{{"added_tokens":[{{"id":{id},"content":"<x>"}}],
                       "model":{{"type":"BPE","vocab":{{"a":0}},"merges":[]}}}}"#
                ),
            ] {
                let file = json::parse(text.as_bytes()).unwrap();
                assert!(Tokenizer::from_json(&file).is_err(), "{text}");
            }
        }
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
