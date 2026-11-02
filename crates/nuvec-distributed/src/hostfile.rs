//! MLX's ring hostfile: a JSON list with one entry per rank, each a list of `"host:port"`
//! addresses (one connection per address), e.g. `[["10.0.0.1:5000"], ["10.0.0.2:5000"]]`.

use nuvec_core::{Error, Result};

use crate::json::{self, Value};

/// The addresses per rank in `text`.
pub(crate) fn parse(text: &str) -> Result<Vec<Vec<String>>> {
    let invalid = || Error::InvalidArgument("hostfile: expected a list of address lists".into());
    let Value::Array(ranks) = json::parse(text, "hostfile")? else {
        return Err(invalid());
    };
    let ranks = ranks
        .into_iter()
        .map(|r| match r {
            Value::Array(addrs) => addrs
                .into_iter()
                .map(|a| match a {
                    Value::String(a) => Ok(a),
                    _ => Err(invalid()),
                })
                .collect::<Result<Vec<_>>>(),
            _ => Err(invalid()),
        })
        .collect::<Result<Vec<_>>>()?;
    if ranks.is_empty() || ranks.iter().any(Vec::is_empty) {
        return Err(Error::InvalidArgument(
            "hostfile: every rank needs at least one address".into(),
        ));
    }
    Ok(ranks)
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
            r#"[["a\q"]]"#,
            "[1]",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
