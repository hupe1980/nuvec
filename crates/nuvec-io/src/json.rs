//! A minimal JSON parser (RFC 8259) for safetensors headers and model files (configs,
//! tokenizers), so that no serialization framework is needed.

use std::collections::BTreeMap;

use nuvec_core::{Error, Result};

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number without fraction or exponent that fits an `i64`.
    Int(i64),
    /// Any other number.
    Float(f64),
    /// A string.
    Str(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, keys in file order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value of `key` in an object (the first, if repeated).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The integer, if this is one.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The number (integers converted), if this is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The boolean, if this is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The entries, if this is an object.
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Self::Object(items) => Some(items),
            _ => None,
        }
    }
}

fn invalid(msg: &str, at: usize) -> Error {
    Error::InvalidArgument(format!("JSON: {msg} at byte {at}"))
}

/// The deepest nesting of arrays and objects accepted (serde_json's default): deeper
/// input is an error instead of a stack overflow.
const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    /// Arrays and objects currently open.
    depth: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(u8::is_ascii_whitespace) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<()> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(invalid(&format!("expected `{}`", c as char), self.i))
        }
    }

    /// Enters the array or object at the current byte.
    fn open(&mut self) -> Result<()> {
        if self.depth == MAX_DEPTH {
            return Err(invalid("nesting too deep", self.i));
        }
        self.depth += 1;
        self.i += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Json> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.open()?;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Object(items));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.eat(b':')?;
                    items.push((key, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Object(items));
                        }
                        _ => return Err(invalid("expected `,` or `}`", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.open()?;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Array(items));
                        }
                        _ => return Err(invalid("expected `,` or `]`", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Null)
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(invalid("value expected", self.i)),
        }
    }

    /// The digits at the current byte (at least one); returns how many.
    fn digits(&mut self, start: usize) -> Result<usize> {
        let from = self.i;
        while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
        }
        match self.i - from {
            0 => Err(invalid("number expected", start)),
            n => Ok(n),
        }
    }

    /// A number in RFC 8259's grammar: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let first = self.i;
        if self.digits(start)? > 1 && self.s[first] == b'0' {
            return Err(invalid("leading zero", start));
        }
        let mut integer = true;
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            self.digits(start)?;
            integer = false;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            self.digits(start)?;
            integer = false;
        }
        // ASCII by construction.
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        match text.parse() {
            Ok(n) if integer => Ok(Json::Int(n)),
            _ => text
                .parse()
                .map(Json::Float)
                .map_err(|_| invalid("number expected", start)),
        }
    }

    fn string(&mut self) -> Result<String> {
        if self.s.get(self.i) != Some(&b'"') {
            return Err(invalid("string expected", self.i));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.s.get(self.i).is_some_and(|&c| c != b'"' && c != b'\\') {
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.s[start..self.i])
                    .map_err(|_| invalid("invalid UTF-8", start))?,
            );
            match self.s.get(self.i) {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let esc = *self
                        .s
                        .get(self.i + 1)
                        .ok_or_else(|| invalid("bad escape", self.i))?;
                    self.i += 2;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let at = self.i - 2;
                            let mut code = self.hex4()?;
                            // A high surrogate must be followed by an escaped low one; a
                            // lone surrogate of either kind is not a character.
                            if (0xD800..0xDC00).contains(&code) {
                                if !self.s[self.i..].starts_with(b"\\u") {
                                    return Err(invalid("lone surrogate", at));
                                }
                                self.i += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(invalid("invalid surrogate pair", at));
                                }
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| invalid("lone surrogate", at))?,
                            );
                        }
                        _ => return Err(invalid("bad escape", self.i - 1)),
                    }
                }
                _ => return Err(invalid("unterminated string", start)),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| invalid("bad \\u escape", self.i))?;
        // Exactly four hex digits: `from_str_radix` alone would accept a sign.
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return Err(invalid("bad \\u escape", self.i));
        }
        self.i += 4;
        std::str::from_utf8(digits)
            .ok()
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| invalid("bad \\u escape", self.i - 4))
    }
}

/// Parses one JSON value (trailing whitespace allowed).
///
/// Strict RFC 8259: lone surrogates, leading zeros and malformed numbers are errors, and
/// arrays and objects nest at most 128 deep (deeper input is an error, not a stack
/// overflow).
pub fn parse(bytes: &[u8]) -> Result<Json> {
    let mut p = Parser {
        s: bytes,
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != bytes.len() {
        return Err(invalid("trailing data", p.i));
    }
    Ok(v)
}

/// `s` as a JSON string literal.
pub(crate) fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A string map as a JSON object.
pub(crate) fn object(map: &BTreeMap<String, String>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("{}:{}", quote(k), quote(v)))
        .collect();
    format!("{{{}}}", items.join(","))
}
