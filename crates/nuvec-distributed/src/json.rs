//! A small JSON reader for MLX's hostfiles and device files (RFC 8259 values; numbers as
//! `f64`).

use nuvec_core::{Error, Result};

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// The value in `text` (`what` names the file in errors).
pub(crate) fn parse(text: &str, what: &str) -> Result<Value> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        at: 0,
        what,
    };
    let v = p.value(0)?;
    p.skip();
    if p.at != p.bytes.len() {
        return Err(p.error("text after the value"));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    what: &'a str,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> Error {
        Error::InvalidArgument(format!("{}: {what} at byte {}", self.what, self.at))
    }

    fn skip(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.bytes.get(self.at).copied()
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(v)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 64 {
            return Err(self.error("nesting deeper than 64"));
        }
        match self.peek() {
            Some(b'n') => self.literal("null", Value::Null),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'"') => self.string().map(Value::String),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(self.error("expected `,` or `]`")),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut fields = Vec::new();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(Value::Object(fields));
                }
                loop {
                    if self.peek() != Some(b'"') {
                        return Err(self.error("expected a key"));
                    }
                    let key = self.string()?;
                    if self.peek() != Some(b':') {
                        return Err(self.error("expected `:`"));
                    }
                    self.at += 1;
                    fields.push((key, self.value(depth + 1)?));
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Value::Object(fields));
                        }
                        _ => return Err(self.error("expected `,` or `}`")),
                    }
                }
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.error("expected a value")),
            None => Err(self.error("unexpected end")),
        }
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|c| matches!(c, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Value::Number)
            .ok_or_else(|| self.error("invalid number"))
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| self.error("invalid \\u escape"))?;
        self.at += 4;
        Ok(digits)
    }

    fn string(&mut self) -> Result<String> {
        self.at += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.at;
            while self
                .bytes
                .get(self.at)
                .is_some_and(|&c| c != b'"' && c != b'\\' && c >= 0x20)
            {
                self.at += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.bytes[start..self.at])
                    .map_err(|_| self.error("invalid UTF-8"))?,
            );
            match self.bytes.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let c = *self
                        .bytes
                        .get(self.at + 1)
                        .ok_or_else(|| self.error("unterminated string"))?;
                    self.at += 2;
                    out.push(match c {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xd800..0xdc00).contains(&code) {
                                if !self.bytes[self.at..].starts_with(b"\\u") {
                                    return Err(self.error("unpaired surrogate"));
                                }
                                self.at += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&low) {
                                    return Err(self.error("unpaired surrogate"));
                                }
                                code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                            }
                            char::from_u32(code).ok_or_else(|| self.error("invalid \\u escape"))?
                        }
                        _ => return Err(self.error("unsupported escape")),
                    });
                }
                Some(_) => return Err(self.error("control character in a string")),
                None => return Err(self.error("unterminated string")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_json_values() {
        let v = parse(
            r#" {"a": [null, true, false, -1.5e2, "x\n\u00e9\ud83d\ude00"], "b": {}} "#,
            "t",
        )
        .unwrap();
        assert_eq!(
            v,
            Value::Object(vec![
                (
                    "a".into(),
                    Value::Array(vec![
                        Value::Null,
                        Value::Bool(true),
                        Value::Bool(false),
                        Value::Number(-150.0),
                        Value::String("x\né😀".into()),
                    ])
                ),
                ("b".into(), Value::Object(vec![])),
            ])
        );
        for bad in [
            "",
            "[",
            "[1,]",
            "{\"a\"}",
            "nul",
            "\"a",
            "\"\\q\"",
            "[1] 2",
            "\"\u{1}\"",
        ] {
            assert!(parse(bad, "t").is_err(), "{bad:?}");
        }
    }
}
