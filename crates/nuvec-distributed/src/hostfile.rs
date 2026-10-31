//! MLX's ring hostfile: a JSON list with one entry per rank, each a list of `"host:port"`
//! addresses (one connection per address), e.g. `[["10.0.0.1:5000"], ["10.0.0.2:5000"]]`.

use nuvec_core::{Error, Result};

/// The addresses per rank in `text`.
pub(crate) fn parse(text: &str) -> Result<Vec<Vec<String>>> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    let ranks = p.list(|p| p.list(Parser::string))?;
    p.skip();
    if p.at != p.bytes.len() {
        return Err(p.error("text after the list"));
    }
    if ranks.is_empty() || ranks.iter().any(Vec::is_empty) {
        return Err(Error::InvalidArgument(
            "hostfile: every rank needs at least one address".into(),
        ));
    }
    Ok(ranks)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> Error {
        Error::InvalidArgument(format!("hostfile: {what} at byte {}", self.at))
    }

    fn skip(&mut self) {
        while self.bytes.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        self.skip();
        if self.bytes.get(self.at) != Some(&c) {
            return Err(self.error(&format!("expected `{}`", c as char)));
        }
        self.at += 1;
        Ok(())
    }

    /// `[item, item, ...]`.
    fn list<T>(&mut self, mut item: impl FnMut(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(items);
        }
        loop {
            items.push(item(self)?);
            self.skip();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(items);
                }
                _ => return Err(self.error("expected `,` or `]`")),
            }
        }
    }

    /// A JSON string (escapes `\"`, `\\`, `\/`).
    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            match self.bytes.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return String::from_utf8(out).map_err(|_| self.error("invalid UTF-8"));
                }
                Some(b'\\') => {
                    let c = *self
                        .bytes
                        .get(self.at + 1)
                        .ok_or_else(|| self.error("unterminated string"))?;
                    if !matches!(c, b'"' | b'\\' | b'/') {
                        return Err(self.error("unsupported escape"));
                    }
                    out.push(c);
                    self.at += 2;
                }
                Some(&c) => {
                    out.push(c);
                    self.at += 1;
                }
                None => return Err(self.error("unterminated string")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_mlx_hostfiles() {
        let text = r#" [ ["10.0.0.1:5000", "10.0.0.1:5001"],
                        ["[::1]:5000", "host.local:5001"] ] "#;
        assert_eq!(
            parse(text).unwrap(),
            [
                vec!["10.0.0.1:5000", "10.0.0.1:5001"],
                vec!["[::1]:5000", "host.local:5001"]
            ]
        );
        for bad in [
            "",
            "[]",
            "[[]]",
            r#"[["a"]"#,
            r#"[["a"]] x"#,
            r#"[["a\n"]]"#,
            "[1]",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
