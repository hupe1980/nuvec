//! Chat templates: a checkpoint's Jinja templates rendered as transformers renders them
//! (`apply_chat_template`), with `minijinja` in transformers' environment:
//! - `trim_blocks` and `lstrip_blocks`, loop controls, Python's string, dict and list
//!   methods;
//! - `tojson` as Python's `json.dumps` (no HTML escaping, Python's float spelling);
//! - `raise_exception`, `strftime_now` (UTC, or [`RenderOptions::now`]), and the
//!   `{% generation %}` tag rendered as its body;
//! - floats, lists and dicts printed as Python prints them;
//! - the named special tokens (`bos_token`, `eos_token`, …) as variables, and reasoning
//!   fields aliased as transformers aliases them.
//!
//! ```no_run
//! use nuvec::io::json;
//! use nuvec_lm::chat::{ChatTemplate, RenderOptions};
//!
//! # fn main() -> nuvec::Result<()> {
//! let dir = std::path::Path::new("models/Qwen3-0.6B-4bit");
//! let template = ChatTemplate::from_dir(dir)?;
//! let tokenizer = nuvec_lm::Tokenizer::from_file(dir.join("tokenizer.json"))?;
//! let messages = json::parse(br#"[{"role": "user", "content": "Hello!"}]"#)?;
//! let options = RenderOptions::new().generation_prompt(true);
//! let text = template.render(messages.as_array().unwrap(), &options)?;
//! let ids = template.encode(&tokenizer, messages.as_array().unwrap(), &options)?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::time::SystemTime;

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Value};
use minijinja::{AutoEscape, Environment, ErrorKind};
use nuvec::io::json::Json;
use nuvec::{Error, Result};

use crate::Tokenizer;
use crate::tokenizer_config::TokenizerConfig;

mod python;

/// transformers' tag for `continue_final_message`.
const CONTINUE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
/// The names a message's reasoning goes under, in transformers' order of lookup.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "thinking", "reasoning"];

/// A checkpoint's chat templates and special tokens, compiled once.
pub struct ChatTemplate {
    env: Environment<'static>,
    names: Vec<String>,
    sources: Vec<String>,
    special_tokens: Vec<(String, String)>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("names", &self.names)
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
    }
}

/// How a conversation is rendered: transformers' `apply_chat_template` arguments.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RenderOptions {
    /// Ends the text with the header of a new assistant message.
    pub add_generation_prompt: bool,
    /// Ends the text inside the final message, for the model to continue it.
    pub continue_final_message: bool,
    /// Tool definitions (a JSON array of schemas).
    pub tools: Option<Json>,
    /// Documents (a JSON array of objects).
    pub documents: Option<Json>,
    /// A template by name, instead of transformers' selection.
    pub template: Option<String>,
    /// The time `strftime_now` reads (UTC); the current time when unset.
    pub now: Option<SystemTime>,
    /// Further template arguments (`enable_thinking`, …), in order; they override the
    /// special tokens.
    pub args: Vec<(String, Json)>,
}

impl RenderOptions {
    /// No generation prompt, no tools, no arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`add_generation_prompt`](Self::add_generation_prompt).
    pub fn generation_prompt(mut self, yes: bool) -> Self {
        self.add_generation_prompt = yes;
        self
    }

    /// Sets [`continue_final_message`](Self::continue_final_message).
    pub fn continue_final_message(mut self, yes: bool) -> Self {
        self.continue_final_message = yes;
        self
    }

    /// Sets the tool definitions.
    pub fn tools(mut self, tools: Json) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the documents.
    pub fn documents(mut self, documents: Json) -> Self {
        self.documents = Some(documents);
        self
    }

    /// Selects a template by name.
    pub fn template(mut self, name: impl Into<String>) -> Self {
        self.template = Some(name.into());
        self
    }

    /// Fixes the time `strftime_now` reads.
    pub fn now(mut self, now: SystemTime) -> Self {
        self.now = Some(now);
        self
    }

    /// Adds (or replaces) a template argument.
    pub fn arg(mut self, name: impl Into<String>, value: Json) -> Self {
        let name = name.into();
        match self.args.iter_mut().find(|(n, _)| *n == name) {
            Some((_, v)) => *v = value,
            None => self.args.push((name, value)),
        }
        self
    }

    /// Whether the argument `name` is given.
    pub fn has_arg(&self, name: &str) -> bool {
        self.args.iter().any(|(n, _)| n == name)
    }
}

impl ChatTemplate {
    /// The templates of the checkpoint in `dir` (see [`TokenizerConfig::from_dir`]); an
    /// error when it has none.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = TokenizerConfig::from_dir(dir)?;
        if config.templates.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "{}: the checkpoint has no chat template",
                dir.display()
            )));
        }
        Self::from_config(&config)
    }

    /// The templates and special tokens of a tokenizer configuration.
    pub fn from_config(config: &TokenizerConfig) -> Result<Self> {
        if config.templates.is_empty() {
            return Err(no_template());
        }
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| Error::InvalidArgument(e.to_string()))?,
        );
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_formatter(|out, _state, value| {
            match python::py_str(value) {
                Some(s) => out.write_str(&s),
                None => write!(out, "{value}"),
            }
            .map_err(|_| minijinja::Error::from(ErrorKind::WriteFailure))
        });
        env.add_filter("tojson", tojson);
        // Jinja2's `string` is Python's `str`; its `iterable` is `iter(x)` succeeding, which
        // `None` and numbers fail (minijinja iterates `none` as empty).
        env.add_filter("string", |v: &Value| match python::py_str(v) {
            Some(s) => Value::from(s),
            None if v.is_undefined() => Value::from(""),
            None => Value::from(v.to_string()),
        });
        env.add_test("iterable", |v: &Value| {
            v.is_undefined()
                || matches!(
                    v.kind(),
                    minijinja::value::ValueKind::String
                        | minijinja::value::ValueKind::Seq
                        | minijinja::value::ValueKind::Map
                        | minijinja::value::ValueKind::Iterable
                )
        });
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_function(
            "strftime_now",
            |state: &minijinja::State, format: String| {
                let now = state
                    .lookup(NOW_VAR)
                    .and_then(|v| v.as_i64())
                    .map_or_else(SystemTime::now, |ms| {
                        SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(ms as u64)
                    });
                python::strftime(&format, now)
            },
        );
        let mut names = Vec::new();
        let mut sources = Vec::new();
        for (name, source) in &config.templates {
            env.add_template_owned(name.clone(), strip_generation_tags(source))
                .map_err(|e| template_error(name, &e))?;
            names.push(name.clone());
            sources.push(source.clone());
        }
        Ok(Self {
            env,
            names,
            sources,
            special_tokens: config.special_tokens.clone(),
        })
    }

    /// A single template (named `default`) with no special tokens.
    pub fn new(source: impl Into<String>) -> Result<Self> {
        Self::from_config(&TokenizerConfig {
            special_tokens: Vec::new(),
            templates: vec![("default".into(), source.into())],
        })
    }

    /// The template names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// The template transformers selects: `options.template` if given; else `tool_use`
    /// when tools are given and it exists; else `default`; a single template always.
    pub fn select(&self, options: &RenderOptions) -> Result<&str> {
        if self.names.len() == 1 && options.template.is_none() {
            return Ok(&self.names[0]);
        }
        let find = |n: &str| self.names.iter().find(|m| *m == n).map(String::as_str);
        let name = match &options.template {
            Some(n) if find(n).is_some() => find(n).unwrap_or_default(),
            Some(n) => {
                return Err(Error::InvalidArgument(format!(
                    "no chat template named {n:?}; the templates are {:?}",
                    self.sorted_names()
                )));
            }
            None if options.tools.is_some() && find("tool_use").is_some() => "tool_use",
            None if find("default").is_some() => "default",
            None => {
                return Err(Error::InvalidArgument(format!(
                    "the checkpoint has several chat templates and no default; name one of {:?}",
                    self.sorted_names()
                )));
            }
        };
        Ok(name)
    }

    fn sorted_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.names().collect();
        names.sort_unstable();
        names
    }

    /// Renders `messages` (JSON objects) as transformers' `apply_chat_template(...,
    /// tokenize=False)` does.
    pub fn render(&self, messages: &[Json], options: &RenderOptions) -> Result<String> {
        if options.continue_final_message && options.add_generation_prompt {
            return Err(Error::InvalidArgument(
                "continue_final_message and add_generation_prompt are not compatible".into(),
            ));
        }
        if messages.is_empty() {
            return Err(Error::InvalidArgument(
                "cannot apply a chat template to an empty conversation".into(),
            ));
        }
        let name = self.select(options)?;
        let mut messages: Vec<Json> = messages.iter().map(alias_reasoning).collect();
        let mut final_text = None;
        if options.continue_final_message {
            final_text = Some(self.tag_final_message(name, &mut messages)?);
        }
        let mut ctx: Vec<(String, Value)> = self
            .special_tokens
            .iter()
            .map(|(n, v)| (n.clone(), Value::from(v.as_str())))
            .collect();
        for (n, v) in &options.args {
            set(&mut ctx, n, value(v));
        }
        set(
            &mut ctx,
            "messages",
            Value::from(messages.iter().map(value).collect::<Vec<_>>()),
        );
        set(
            &mut ctx,
            "tools",
            options.tools.as_ref().map_or(Value::from(()), value),
        );
        set(
            &mut ctx,
            "documents",
            options.documents.as_ref().map_or(Value::from(()), value),
        );
        set(
            &mut ctx,
            "add_generation_prompt",
            Value::from(options.add_generation_prompt),
        );
        if let Some(now) = options.now {
            let ms = now
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as i64);
            set(&mut ctx, NOW_VAR, Value::from(ms));
        }
        let template = self
            .env
            .get_template(name)
            .map_err(|e| template_error(name, &e))?;
        let mut text = template
            .render(Value::from_pairs(ctx))
            .map_err(|e| template_error(name, &e))?;
        if let Some(final_text) = final_text {
            text = cut_continued(text, &final_text)?;
        }
        Ok(text)
    }

    /// The ids transformers' `apply_chat_template(..., tokenize=True)` returns: the
    /// rendered text encoded without the tokenizer's added special tokens.
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        messages: &[Json],
        options: &RenderOptions,
    ) -> Result<Vec<u32>> {
        Ok(tokenizer.encode_text(&self.render(messages, options)?))
    }

    /// Appends transformers' tag to the final message's content (or its last text part)
    /// and returns the original text.
    fn tag_final_message(&self, name: &str, messages: &mut [Json]) -> Result<String> {
        let source = &self.sources[self.names.iter().position(|n| n == name).unwrap_or(0)];
        if !source.contains("content") {
            return Err(Error::InvalidArgument(
                "continue_final_message is set to \"content\" but this is not an accepted \
                 field in the chat_template"
                    .into(),
            ));
        }
        let missing = || {
            Error::InvalidArgument(
                "continue_final_message is set but the final message has no \"content\" to \
                 continue"
                    .into(),
            )
        };
        let last = messages.last_mut().ok_or_else(missing)?;
        let Json::Object(fields) = last else {
            return Err(missing());
        };
        let content = fields
            .iter_mut()
            .find(|(k, _)| k == "content")
            .map(|(_, v)| v)
            .ok_or_else(missing)?;
        match content {
            Json::Str(s) => {
                let original = s.clone();
                s.push_str(CONTINUE_TAG);
                Ok(original)
            }
            Json::Array(parts) => {
                for part in parts.iter_mut().rev() {
                    if let Json::Object(p) = part
                        && let Some((_, Json::Str(t))) = p.iter_mut().find(|(k, _)| k == "text")
                    {
                        let original = t.clone();
                        t.push_str(CONTINUE_TAG);
                        return Ok(original);
                    }
                }
                Err(Error::InvalidArgument(
                    "continue_final_message is set but we could not find any text to continue \
                     in the final message"
                        .into(),
                ))
            }
            Json::Null => Err(missing()),
            _ => Err(missing()),
        }
    }
}

/// The context key the fixed time travels under (not a name templates use).
const NOW_VAR: &str = "__nuvec_now_ms";

fn no_template() -> Error {
    Error::InvalidArgument("the checkpoint has no chat template".into())
}

fn set(ctx: &mut Vec<(String, Value)>, name: &str, v: Value) {
    match ctx.iter_mut().find(|(n, _)| n == name) {
        Some((_, old)) => *old = v,
        None => ctx.push((name.to_owned(), v)),
    }
}

/// transformers' cut after rendering with the tag: at the tag's last occurrence, trimming
/// trailing space the template removed.
fn cut_continued(text: String, final_text: &str) -> Result<String> {
    let tag = CONTINUE_TAG.trim();
    if !text.contains(final_text.trim()) || !text.contains(tag) {
        return Err(Error::InvalidArgument(
            "continue_final_message is set but the final message does not appear in the chat \
             after applying the chat template"
                .into(),
        ));
    }
    let at = text.rfind(tag).unwrap_or(text.len());
    if text[at..].starts_with(CONTINUE_TAG) {
        Ok(text[..at].to_owned())
    } else {
        Ok(text[..at].trim_end().to_owned())
    }
}

/// transformers 5.22's `_alias_reasoning_fields`: a message with reasoning under any of
/// [`REASONING_FIELDS`] becomes `{"thinking": r, **message, "reasoning_content": r}`.
fn alias_reasoning(message: &Json) -> Json {
    let Json::Object(fields) = message else {
        return message.clone();
    };
    let reasoning = REASONING_FIELDS.iter().find_map(|f| {
        fields
            .iter()
            .find(|(k, v)| k == f && !matches!(v, Json::Null))
            .map(|(_, v)| v.clone())
    });
    let Some(r) = reasoning else {
        return message.clone();
    };
    let mut out: Vec<(String, Json)> = vec![("thinking".into(), r.clone())];
    for (k, v) in fields {
        match out.iter_mut().find(|(n, _)| n == k) {
            Some((_, old)) => *old = v.clone(),
            None => out.push((k.clone(), v.clone())),
        }
    }
    match out.iter_mut().find(|(n, _)| n == "reasoning_content") {
        Some((_, old)) => *old = r,
        None => out.push(("reasoning_content".into(), r)),
    }
    Json::Object(out)
}

/// A JSON value as a template value: objects keep their key order.
fn value(json: &Json) -> Value {
    match json {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Int(i) => Value::from(*i),
        Json::Float(f) => Value::from(*f),
        Json::Str(s) => Value::from(s.as_str()),
        Json::Array(items) => Value::from(items.iter().map(value).collect::<Vec<_>>()),
        Json::Object(fields) => {
            Value::from_pairs(fields.iter().map(|(k, v)| (k.as_str(), value(v))))
        }
    }
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`.
fn tojson(v: &Value, kwargs: Kwargs) -> std::result::Result<Value, minijinja::Error> {
    let mut o = python::Dumps::default();
    if let Some(e) = kwargs.get::<Option<bool>>("ensure_ascii")? {
        o.ensure_ascii = e;
    }
    let indent: Option<Value> = kwargs.get("indent")?;
    if let Some(indent) = indent.filter(|i| !i.is_none()) {
        o.indent = Some(match indent.as_i64() {
            Some(n) => " ".repeat(n.max(0) as usize),
            None => indent.to_string(),
        });
        o.item_separator = ",".into();
    }
    let separators: Option<Value> = kwargs.get("separators")?;
    if let Some(s) = separators.filter(|s| !s.is_none()) {
        let item = s.get_item_by_index(0)?;
        let key = s.get_item_by_index(1)?;
        o.item_separator = item.to_string();
        o.key_separator = key.to_string();
    }
    if let Some(sort) = kwargs.get::<Option<bool>>("sort_keys")? {
        o.sort_keys = sort;
    }
    kwargs.assert_all_used()?;
    Ok(Value::from(python::dumps(v, &o)?))
}

/// Replaces transformers' `{% generation %}…{% endgeneration %}` (its `AssistantTracker`
/// extension, which renders the body) with `{% if true %}…{% endif %}`, keeping the tags'
/// whitespace control.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(end) = tag.find("%}") else {
            out.push_str(tag);
            return out;
        };
        let inner = &tag[2..end];
        let (open_trim, body) = match inner.strip_prefix(['-', '+']) {
            Some(b) => (&inner[..1], b),
            None => ("", inner),
        };
        let (body, close_trim) = match body.strip_suffix(['-', '+']) {
            Some(b) => (b, &body[body.len() - 1..]),
            None => (body, ""),
        };
        let replacement = match body.trim() {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        match replacement {
            Some(r) => out.push_str(&format!("{{%{open_trim} {r} {close_trim}%}}")),
            None => out.push_str(&tag[..end + 2]),
        }
        rest = &tag[end + 2..];
    }
    out.push_str(rest);
    out
}

fn template_error(name: &str, e: &minijinja::Error) -> Error {
    let message = match e.detail() {
        Some(d) => d.to_owned(),
        None => e.kind().to_string(),
    };
    match e.line() {
        Some(line) if e.kind() != ErrorKind::InvalidOperation => {
            Error::InvalidArgument(format!("chat template {name}: {message} (line {line})"))
        }
        _ => Error::InvalidArgument(format!("chat template {name}: {message}")),
    }
}

/// mlx-lm's `has_thinking`: whether the vocabulary has a thinking token pair (`<think>`
/// and `</think>`, `<longcat_think>`, `<|think:start|>`), Gemma 4's `<|channel>` and
/// `<channel|>`, or the "xtml" markers. mlx-lm passes it as every rendering's default
/// `enable_thinking`.
pub fn has_thinking(tokenizer: &Tokenizer) -> bool {
    let has = |t: &str| tokenizer.token_id(t).is_some();
    [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
        ("<|think:start|>", "<|think:end|>"),
        ("<|channel>", "<channel|>"),
    ]
    .iter()
    .any(|(a, b)| has(a) && has(b))
        || ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"]
            .iter()
            .all(|t| has(t))
}

/// `options` with mlx-lm's default `enable_thinking` ([`has_thinking`]) unless given, as
/// mlx-lm's tokenizer wrapper renders.
pub fn mlx_lm_options(tokenizer: &Tokenizer, options: RenderOptions) -> RenderOptions {
    if options.has_arg("enable_thinking") {
        options
    } else {
        options.arg("enable_thinking", Json::Bool(has_thinking(tokenizer)))
    }
}

/// `mlx_lm.generate`'s prompt ids. With a template: `[system] + [user: prompt] +
/// [assistant: prefill]` rendered with a generation prompt (or continuing the prefill),
/// with `options`' arguments and time and mlx-lm's default `enable_thinking`, encoded
/// without special tokens. Without one: the prompt encoded with them.
pub fn prompt_ids(
    template: Option<&ChatTemplate>,
    tokenizer: &Tokenizer,
    prompt: &str,
    system: Option<&str>,
    prefill: Option<&str>,
    options: RenderOptions,
) -> Result<Vec<u32>> {
    let Some(template) = template else {
        return Ok(tokenizer.encode(prompt));
    };
    let message = |role: &str, content: &str| {
        Json::Object(vec![
            ("role".into(), Json::Str(role.into())),
            ("content".into(), Json::Str(content.into())),
        ])
    };
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(message("system", s));
    }
    messages.push(message("user", prompt));
    if let Some(p) = prefill {
        messages.push(message("assistant", p));
    }
    let options = options
        .generation_prompt(prefill.is_none())
        .continue_final_message(prefill.is_some());
    template.encode(tokenizer, &messages, &mlx_lm_options(tokenizer, options))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuvec::io::json;

    fn messages(s: &str) -> Vec<Json> {
        json::parse(s.as_bytes())
            .unwrap()
            .as_array()
            .unwrap()
            .to_vec()
    }

    fn render(t: &str, m: &str, o: &RenderOptions) -> Result<String> {
        ChatTemplate::new(t).unwrap().render(&messages(m), o)
    }

    #[test]
    fn renders_in_transformers_environment() {
        let o = RenderOptions::new();
        let user = r#"[{"role": "user", "content": " Hi "}]"#;
        // trim_blocks and lstrip_blocks; Python methods; loop controls and previtem.
        let t =
            "{% for m in messages %}\n  {{ m.role|upper }}:{{ m.content.strip() }}\n{% endfor %}";
        // lstrip_blocks strips before block tags only, as Jinja2.
        assert_eq!(render(t, user, &o).unwrap(), "  USER:Hi\n");
        let t = "{% for x in [1, 2, 3] %}{% if x == 3 %}{% break %}{% endif %}\
                 {{ loop.previtem }}{% endfor %}";
        assert_eq!(render(t, user, &o).unwrap(), "1");
        let t = "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}\
                 {% endfor %}{{ ns.n }}";
        assert_eq!(render(t, user, &o).unwrap(), "1");
        // Python printing and json.dumps.
        let t =
            "{{ [1, 'a', true, none, 1e-05] }} {{ {'k': 0.5}|tojson }} {{ 'x'|tojson(indent=2) }}";
        assert_eq!(
            render(t, user, &o).unwrap(),
            r#"[1, 'a', True, None, 1e-05] {"k": 0.5} "x""#
        );
        assert_eq!(
            render(
                "{{ [0.5, 1e-05]|string }} {{ none is iterable }} {{ x is iterable }} {{ 'a' is iterable }}",
                user,
                &o
            )
            .unwrap(),
            "[0.5, 1e-05] False True True"
        );
        assert_eq!(
            render("{{ messages|tojson(indent=1) }}", user, &o).unwrap(),
            "[\n {\n  \"role\": \"user\",\n  \"content\": \" Hi \"\n }\n]"
        );
        // tools and documents are defined and None when absent.
        assert_eq!(
            render(
                "{{ tools is none }} {{ documents is defined }} {{ add_generation_prompt }}",
                user,
                &o
            )
            .unwrap(),
            "True True False"
        );
        // {% generation %} renders its body, keeping whitespace control.
        let t = "a  {%- generation -%}  b  {%- endgeneration %}\nc";
        assert_eq!(render(t, user, &o).unwrap(), "abc");
        // strftime_now reads the fixed time.
        let o = RenderOptions::new().now(SystemTime::UNIX_EPOCH);
        assert_eq!(
            render("{{ strftime_now('%d %b %Y') }}", user, &o).unwrap(),
            "01 Jan 1970"
        );
    }

    #[test]
    fn errors_name_the_template_and_message() {
        let user = r#"[{"role": "user", "content": "Hi"}]"#;
        let o = RenderOptions::new();
        let e = render("{{ raise_exception('roles must alternate') }}", user, &o).unwrap_err();
        assert_eq!(
            e.to_string(),
            "invalid argument: chat template default: roles must alternate"
        );
        assert!(ChatTemplate::new("{% if %}").is_err());
        assert!(render("{{ messages[0].content + [1] }}", user, &o).is_err());
        let o = RenderOptions::new()
            .generation_prompt(true)
            .continue_final_message(true);
        assert!(render("{{ messages }}", user, &o).is_err());
        assert!(render("{{ messages }}", "[]", &RenderOptions::new()).is_err());
        assert!(ChatTemplate::from_config(&TokenizerConfig::default()).is_err());
    }

    #[test]
    fn selects_templates_as_transformers() {
        let config = TokenizerConfig {
            special_tokens: vec![("bos_token".into(), "<s>".into())],
            templates: vec![
                ("default".into(), "D{{ bos_token }}".into()),
                ("tool_use".into(), "T".into()),
            ],
        };
        let t = ChatTemplate::from_config(&config).unwrap();
        let m = messages(r#"[{"role": "user", "content": "Hi"}]"#);
        assert_eq!(t.render(&m, &RenderOptions::new()).unwrap(), "D<s>");
        let tools = json::parse(b"[]").unwrap();
        assert_eq!(
            t.render(&m, &RenderOptions::new().tools(tools)).unwrap(),
            "T"
        );
        assert_eq!(
            t.render(&m, &RenderOptions::new().template("tool_use"))
                .unwrap(),
            "T"
        );
        let e = t
            .render(&m, &RenderOptions::new().template("x"))
            .unwrap_err();
        assert!(e.to_string().contains(r#"["default", "tool_use"]"#), "{e}");
        // Arguments override the special tokens.
        let o = RenderOptions::new().arg("bos_token", Json::Str("B".into()));
        assert_eq!(t.render(&m, &o).unwrap(), "DB");
    }

    #[test]
    fn continues_the_final_message_and_aliases_reasoning() {
        let t = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}";
        let m = r#"[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hel"}]"#;
        let o = RenderOptions::new().continue_final_message(true);
        assert_eq!(render(t, m, &o).unwrap(), "<user>Hi</user><assistant>Hel");
        let t = "{% for m in messages %}{{ m.content|trim }}|{% endfor %}";
        assert_eq!(render(t, m, &o).unwrap(), "Hi|Hel");

        let m = r#"[{"role": "assistant", "content": "4", "reasoning": "2+2"}]"#;
        let t = "{% for m in messages %}{{ m|tojson }}{% endfor %}";
        assert_eq!(
            render(t, m, &RenderOptions::new()).unwrap(),
            r#"{"thinking": "2+2", "role": "assistant", "content": "4", "reasoning": "2+2", "reasoning_content": "2+2"}"#
        );
    }
}
