//! `tokenizer_config.json`: the special-token strings and the chat templates, read as
//! transformers reads them.

use std::path::Path;

use nuvec::io::json::{self, Json};
use nuvec::{Error, Result};

/// transformers' named special tokens (`SPECIAL_TOKENS_ATTRIBUTES`), in its order.
const NAMES: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// A checkpoint's tokenizer configuration: the named special tokens and the chat templates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenizerConfig {
    /// The named special tokens present, as `(name, content)` in transformers' order:
    /// `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`,
    /// `mask_token`, then the model-specific ones (`image_token`, …).
    pub special_tokens: Vec<(String, String)>,
    /// The chat templates as `(name, source)`. A single template is named `default`.
    pub templates: Vec<(String, String)>,
}

impl TokenizerConfig {
    /// Reads `dir`'s configuration as transformers loads it:
    /// - the special tokens from `tokenizer_config.json` (strings or `{"content": ...}`);
    ///   `special_tokens_map.json` overrides them only when the configuration has no
    ///   `added_tokens_decoder`;
    /// - the templates from `tokenizer_config.json`'s `chat_template` (a string, or a list
    ///   of `{name, template}`), replaced by `chat_template.jinja` (`default`) and
    ///   `additional_chat_templates/*.jinja` (named by file stem) when those exist.
    ///
    /// Missing files give the defaults.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Json::Null);
        let mut tokens: Vec<(String, Option<String>)> = NAMES
            .iter()
            .map(|&n| (n.to_owned(), config.get(n).and_then(content)))
            .collect();
        if config.get("added_tokens_decoder").is_none()
            && let Some(map) = read_json(&dir.join("special_tokens_map.json"))?
        {
            for (name, value) in &mut tokens {
                if let Some(v) = map.get(name) {
                    *value = content(v);
                }
            }
        }
        // transformers 5's model-specific special tokens, named after the standard ones:
        // the configuration's other `*_token` strings (in file order) and an
        // `extra_special_tokens` mapping, else its `model_specific_special_tokens`.
        let mut specific: Vec<(String, Option<String>)> = Vec::new();
        fn add(specific: &mut Vec<(String, Option<String>)>, name: &str, value: Option<String>) {
            match specific.iter_mut().find(|(n, _)| n == name) {
                Some((_, v)) => *v = value,
                None => specific.push((name.to_owned(), value)),
            }
        }
        for (key, value) in config.as_object().unwrap_or_default() {
            if key.ends_with("_token")
                && !NAMES.contains(&key.as_str())
                && let v @ Some(_) = content(value)
            {
                add(&mut specific, key, v);
            }
        }
        if let Some(Json::Object(extra)) = config.get("extra_special_tokens") {
            for (key, value) in extra {
                add(&mut specific, key, content(value));
            }
        }
        if specific.is_empty()
            && let Some(Json::Object(explicit)) = config.get("model_specific_special_tokens")
        {
            for (key, value) in explicit {
                add(&mut specific, key, content(value));
            }
        }
        tokens.extend(specific);
        let mut templates = match config.get("chat_template") {
            Some(Json::Str(s)) => vec![("default".to_owned(), s.clone())],
            Some(Json::Array(list)) => list
                .iter()
                .map(|t| match (t.get("name"), t.get("template")) {
                    (Some(Json::Str(n)), Some(Json::Str(s))) => Ok((n.clone(), s.clone())),
                    _ => Err(Error::InvalidArgument(format!(
                        "{}: a chat_template entry needs a name and a template",
                        dir.join("tokenizer_config.json").display()
                    ))),
                })
                .collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        let mut files = Vec::new();
        if let Some(s) = read_text(&dir.join("chat_template.jinja"))? {
            files.push(("default".to_owned(), s));
        }
        if let Ok(entries) = std::fs::read_dir(dir.join("additional_chat_templates")) {
            let mut named: Vec<_> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "jinja"))
                .collect();
            named.sort();
            for path in named {
                let name = path.file_stem().unwrap_or_default().to_string_lossy();
                if let Some(s) = read_text(&path)? {
                    files.push((name.into_owned(), s));
                }
            }
        }
        if !files.is_empty() {
            templates = files;
        }
        Ok(Self {
            special_tokens: tokens
                .into_iter()
                .filter_map(|(n, v)| v.map(|v| (n, v)))
                .collect(),
            templates,
        })
    }

    /// The named special token `name` (`"eos_token"`, …), if present.
    pub fn special_token(&self, name: &str) -> Option<&str> {
        self.special_tokens
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A special token's content: a string, or an `AddedToken` object's `content`.
fn content(value: &Json) -> Option<String> {
    match value {
        Json::Str(s) => Some(s.clone()),
        Json::Object(_) => value
            .get("content")
            .and_then(Json::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn read_text(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Io(format!("{}: {e}", path.display()))),
    }
}

fn read_json(path: &Path) -> Result<Option<Json>> {
    read_text(path)?
        .map(|s| {
            json::parse(s.as_bytes()).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(files: &[(&str, &str)]) -> tempdir::Dir {
        let d = tempdir::Dir::new();
        for (name, text) in files {
            let path = d.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        d
    }

    mod tempdir {
        pub(super) struct Dir(pub(super) std::path::PathBuf);
        impl Dir {
            pub(super) fn new() -> Self {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let p = std::env::temp_dir().join(format!(
                    "nuvec-tokcfg-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn reads_special_tokens_and_templates_as_transformers() {
        let d = dir(&[(
            "tokenizer_config.json",
            r#"{"bos_token": {"content": "<s>", "lstrip": false}, "eos_token": "</s>",
                "pad_token": null, "chat_template": "{{ bos_token }}"}"#,
        )]);
        let c = TokenizerConfig::from_dir(&d.0).unwrap();
        assert_eq!(
            c.special_tokens,
            [
                ("bos_token".into(), "<s>".into()),
                ("eos_token".into(), "</s>".into())
            ]
        );
        assert_eq!(c.special_token("eos_token"), Some("</s>"));
        assert_eq!(c.templates, [("default".into(), "{{ bos_token }}".into())]);

        // A named list; `special_tokens_map.json` overrides without `added_tokens_decoder`.
        let d = dir(&[
            (
                "tokenizer_config.json",
                r#"{"eos_token": "a", "chat_template": [{"name": "default", "template": "d"},
                    {"name": "tool_use", "template": "t"}]}"#,
            ),
            (
                "special_tokens_map.json",
                r#"{"eos_token": {"content": "b"}}"#,
            ),
        ]);
        let c = TokenizerConfig::from_dir(&d.0).unwrap();
        assert_eq!(c.special_token("eos_token"), Some("b"));
        assert_eq!(c.templates.len(), 2);
        assert_eq!(c.templates[1], ("tool_use".into(), "t".into()));

        // With `added_tokens_decoder`, the map is not read; `.jinja` files win.
        let d = dir(&[
            (
                "tokenizer_config.json",
                r#"{"eos_token": "a", "added_tokens_decoder": {}, "chat_template": "old"}"#,
            ),
            ("special_tokens_map.json", r#"{"eos_token": "b"}"#),
            ("chat_template.jinja", "new"),
            ("additional_chat_templates/tool_use.jinja", "tools"),
        ]);
        let c = TokenizerConfig::from_dir(&d.0).unwrap();
        assert_eq!(c.special_token("eos_token"), Some("a"));
        assert_eq!(
            c.templates,
            [
                ("default".into(), "new".into()),
                ("tool_use".into(), "tools".into())
            ]
        );

        // No files: the defaults.
        let d = dir(&[]);
        assert_eq!(
            TokenizerConfig::from_dir(&d.0).unwrap(),
            TokenizerConfig::default()
        );
    }
}
