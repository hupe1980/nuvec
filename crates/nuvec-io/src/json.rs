//! The JSON subset of safetensors headers: objects, arrays, strings, integers, booleans and
//! null.

use std::collections::BTreeMap;

use nuvec_core::{Error, Result};

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// Integers only (safetensors headers hold no other numbers).
    Int(i64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in file order.
    Object(Vec<(String, Json)>),
}

fn invalid(msg: &str, at: usize) -> Error {
    Error::InvalidArgument(format!("safetensors header: {msg} at byte {at}"))
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
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i])
                    .ok()
                    .and_then(|t| t.parse().ok())
                    .map(Json::Int)
                    .ok_or_else(|| invalid("integer expected", start))
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
pub(crate) fn parse(bytes: &[u8]) -> Result<Json> {
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
