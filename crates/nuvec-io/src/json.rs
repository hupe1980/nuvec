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

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
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

    fn value(&mut self) -> Result<Json> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
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
                            return Ok(Json::Object(items));
                        }
                        _ => return Err(invalid("expected `,` or `}`", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
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
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                self.i += 1;
                while self
                    .s
                    .get(self.i)
                    .is_some_and(|c| c.is_ascii_digit() || b".eE+-".contains(c))
                {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                text.parse()
                    .map(Json::Int)
                    .or_else(|_| text.parse().map(Json::Float))
                    .map_err(|_| invalid("number expected", start))
            }
            _ => Err(invalid("value expected", self.i)),
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
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                self.i += 2;
                                let low = self.hex4()?;
                                code = 0x10000
                                    + ((code - 0xD800) << 10)
                                    + (low.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
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
        self.i += 4;
        std::str::from_utf8(digits)
            .ok()
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| invalid("bad \\u escape", self.i - 4))
    }
}

/// Parses one JSON value (trailing whitespace allowed).
pub fn parse(bytes: &[u8]) -> Result<Json> {
    let mut p = Parser { s: bytes, i: 0 };
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
