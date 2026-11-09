//! Chat templates and chat training data against transformers 5.22.1 and mlx-lm 0.32.2
//! (`scripts/chat_fixtures.py`). Templates and tokenizers come from the Hugging Face cache;
//! a checkpoint that is not there is skipped with a message.
#![cfg(feature = "chat")]

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use nuvec::io::json::{self, Json};
use nuvec_lm::chat::{self, ChatTemplate, RenderOptions};
use nuvec_lm::tuner::{self, DataOptions, Dataset};
use nuvec_lm::{Tokenizer, TokenizerConfig};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat");
/// The recorders' `strftime_now`: 2026-11-08 12:00 UTC.
const NOW: u64 = 1_794_139_200;

fn read(path: impl AsRef<Path>) -> Json {
    let path = path.as_ref();
    json::parse(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

fn snapshot(repo: &str) -> Option<PathBuf> {
    match nuvec_lm::resolve(repo) {
        Ok(dir) if dir.join("tokenizer.json").exists() => Some(dir),
        _ => {
            eprintln!(
                "skipped {repo}: not in the Hugging Face cache (uv run scripts/chat_fixtures.py corpus)"
            );
            None
        }
    }
}

fn ids(value: &Json) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as u32)
        .collect()
}

/// The first differing position of two strings, with some context.
fn first_difference(a: &str, b: &str) -> String {
    let at = a
        .char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map_or(a.len().min(b.len()), |((i, _), _)| i);
    let from = a[..at].char_indices().rev().nth(40).map_or(0, |(i, _)| i);
    format!(
        "at byte {at}: nuvec {:?} | transformers {:?}",
        a.get(from..(at + 40).min(a.len())).unwrap_or(""),
        b.get(from..(at + 40).min(b.len())).unwrap_or("")
    )
}

#[test]
fn corpus_renders_as_transformers() {
    let corpus = read(format!("{FIXTURES}/corpus.json"));
    let conversations = read(format!("{FIXTURES}/conversations.json"));
    let waivers = std::fs::read(format!("{FIXTURES}/waivers.json"))
        .ok()
        .map(|b| json::parse(&b).unwrap());
    let waived = |repo: &str, case: &str| {
        waivers.as_ref().is_some_and(|w| {
            w.as_array().unwrap().iter().any(|x| {
                x.get("repo").and_then(Json::as_str) == Some(repo)
                    && x.get("case").and_then(Json::as_str) == Some(case)
            })
        })
    };
    let (mut total, mut failures, mut waived_cases, mut checked_repos) = (0, Vec::new(), 0, 0);
    for entry in corpus.get("repos").unwrap().as_array().unwrap() {
        let repo = entry.get("repo").unwrap().as_str().unwrap();
        let Some(dir) = snapshot(repo) else {
            continue;
        };
        checked_repos += 1;
        let fixture = read(format!("{FIXTURES}/{}.json", repo.replace('/', "--")));
        let config = TokenizerConfig::from_dir(&dir).unwrap();
        let special: Vec<(String, String)> = fixture
            .get("special_tokens")
            .unwrap()
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        assert_eq!(config.special_tokens, special, "{repo}: special tokens");
        let template = ChatTemplate::from_config(&config).unwrap();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"));
        if let Err(e) = &tokenizer {
            eprintln!("{repo}: ids not checked, the tokenizer does not load: {e}");
        }
        for case in fixture.get("cases").unwrap().as_array().unwrap() {
            total += 1;
            let name = case.get("conversation").unwrap().as_str().unwrap();
            let flag = |k: &str| case.get(k).and_then(Json::as_bool);
            let id = format!(
                "{name} gen={:?} thinking={:?} continue={:?}",
                flag("add_generation_prompt"),
                flag("enable_thinking"),
                flag("continue_final_message")
            );
            let conv = conversations.get(name).unwrap();
            let mut options = RenderOptions::new()
                .generation_prompt(flag("add_generation_prompt").unwrap_or(false))
                .continue_final_message(flag("continue_final_message").unwrap_or(false))
                .now(SystemTime::UNIX_EPOCH + Duration::from_secs(NOW));
            if let Some(tools) = conv.get("tools") {
                options = options.tools(tools.clone());
            }
            if let Some(t) = flag("enable_thinking") {
                options = options.arg("enable_thinking", Json::Bool(t));
            }
            let messages = conv.get("messages").unwrap().as_array().unwrap();
            let got = template.render(messages, &options);
            let problem = match (&got, case.get("text"), case.get("error")) {
                (Ok(text), Some(Json::Str(want)), _) if text == want => None,
                (Ok(text), Some(Json::Str(want)), _) => Some(first_difference(text, want)),
                (Err(_), None, Some(_)) => None,
                (Ok(text), None, Some(e)) => Some(format!(
                    "transformers fails ({}), nuvec renders {:?}",
                    e.as_str().unwrap(),
                    &text[..text.len().min(80)]
                )),
                (Err(e), _, _) => Some(format!("nuvec fails: {e}")),
                _ => Some("bad fixture".into()),
            };
            let problem = problem.or_else(|| {
                let (Ok(text), Some(want), Ok(tokenizer)) = (&got, case.get("ids"), &tokenizer)
                else {
                    return None;
                };
                let have = tokenizer.encode_text(text);
                (have != ids(want)).then(|| "ids differ (tokenizer)".to_owned())
            });
            if let Some(p) = problem {
                if waived(repo, &id) {
                    waived_cases += 1;
                    eprintln!("waived {repo} {id}: {p}");
                } else {
                    failures.push(format!("{repo} {id}: {p}"));
                }
            }
        }
    }
    eprintln!("{checked_repos} repos, {total} cases, {waived_cases} waived");
    assert!(
        failures.is_empty(),
        "{} of {total} cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        waived_cases * 50 <= total,
        "{waived_cases} of {total} cases waived (more than 2%)"
    );
}

#[test]
fn datasets_equal_mlx_lm() {
    let rows = read(format!("{FIXTURES}/datasets/rows.json"));
    let mut checked = 0;
    for entry in std::fs::read_dir(format!("{FIXTURES}/datasets")).unwrap() {
        let path = entry.unwrap().path();
        if path.file_name().unwrap() == "rows.json" {
            continue;
        }
        let fixture = read(&path);
        let repo = fixture.get("repo").unwrap().as_str().unwrap();
        let Some(dir) = snapshot(repo) else {
            continue;
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let config = TokenizerConfig::from_dir(&dir).unwrap();
        let model = nuvec_lm::Config::from_file(dir.join("config.json")).unwrap();
        let eos = tuner::eos_token_id(&tokenizer, &config, &model).unwrap();
        assert_eq!(
            i64::from(eos),
            fixture.get("eos_token_id").unwrap().as_i64().unwrap(),
            "{repo}: EOS"
        );
        assert_eq!(
            chat::has_thinking(&tokenizer),
            fixture.get("has_thinking").unwrap().as_bool().unwrap(),
            "{repo}: has_thinking"
        );
        let template = ChatTemplate::from_dir(&dir).ok();
        for set in fixture.get("sets").unwrap().as_array().unwrap() {
            let name = set.get("rows").unwrap().as_str().unwrap();
            let key = |k: &str, default: &str| {
                set.get(k)
                    .and_then(Json::as_str)
                    .unwrap_or(default)
                    .to_owned()
            };
            let options = DataOptions {
                mask_prompt: set.get("mask_prompt").unwrap().as_bool().unwrap(),
                text_feature: key("text_feature", "text"),
                chat_feature: key("chat_feature", "messages"),
                prompt_feature: key("prompt_feature", "prompt"),
                completion_feature: key("completion_feature", "completion"),
                now: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(NOW)),
            };
            let data = rows.get(name).unwrap().as_array().unwrap();
            let got = Dataset::from_rows(data, &tokenizer, template.as_ref(), eos, &options);
            let id = format!("{repo} {name} mask_prompt={}", options.mask_prompt);
            match (got, set.get("out")) {
                (Ok(ds), Some(want)) => {
                    let want = want.as_array().unwrap();
                    assert_eq!(ds.len(), want.len(), "{id}");
                    for (i, ((tokens, offset), w)) in ds.rows().zip(want).enumerate() {
                        let w = w.as_array().unwrap();
                        assert_eq!(tokens, ids(&w[0]).as_slice(), "{id} row {i}: tokens");
                        assert_eq!(
                            offset as i64,
                            w[1].as_i64().unwrap(),
                            "{id} row {i}: offset"
                        );
                    }
                }
                (Err(e), None) => {
                    let want = set.get("error").unwrap().as_str().unwrap();
                    assert!(e.to_string().contains(want), "{id}: {e} (mlx-lm: {want})");
                }
                (Ok(_), None) => panic!("{id}: mlx-lm fails, nuvec loads"),
                (Err(e), Some(_)) => panic!("{id}: {e}"),
            }
            checked += 1;
        }
    }
    eprintln!("{checked} dataset cases");
}

#[test]
fn generate_prompts_equal_mlx_lm() {
    let fixture = read(format!("{FIXTURES}/generate.json"));
    let prompts = fixture.get("prompts").unwrap().as_array().unwrap();
    for entry in fixture.get("repos").unwrap().as_array().unwrap() {
        let repo = entry.get("repo").unwrap().as_str().unwrap();
        let Some(dir) = snapshot(repo) else {
            continue;
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let template = ChatTemplate::from_dir(&dir).ok();
        for (p, want) in prompts
            .iter()
            .zip(entry.get("ids").unwrap().as_array().unwrap())
        {
            let text = |k: &str| p.get(k).and_then(Json::as_str);
            let mut options =
                RenderOptions::new().now(SystemTime::UNIX_EPOCH + Duration::from_secs(NOW));
            for (k, v) in p
                .get("chat_template_config")
                .and_then(Json::as_object)
                .unwrap_or_default()
            {
                options = options.arg(k.clone(), v.clone());
            }
            let ignore = p.get("ignore_chat_template").is_some();
            let got = chat::prompt_ids(
                if ignore { None } else { template.as_ref() },
                &tokenizer,
                text("prompt").unwrap(),
                text("system_prompt"),
                text("prefill_response"),
                options,
            );
            match (got, want.get("error")) {
                (Ok(ids_), None) => assert_eq!(ids_, ids(want), "{repo} {p:?}"),
                (Err(_), Some(_)) => {}
                (got, _) => panic!("{repo} {p:?}: {got:?}, mlx-lm {want:?}"),
            }
        }
    }
}

/// The load time of `scripts/chat_fixtures.py timing --write DIR`'s rows with
/// `mask_prompt` (Qwen3-0.6B-4bit), best of 3:
/// `NUVEC_CHAT_ROWS=DIR cargo test --release -p nuvec-lm --test chat -- --ignored load_time`.
#[test]
#[ignore = "a measurement"]
fn load_time() {
    let rows = PathBuf::from(std::env::var("NUVEC_CHAT_ROWS").expect("NUVEC_CHAT_ROWS=DIR"));
    let dir = snapshot("mlx-community/Qwen3-0.6B-4bit").unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let template = ChatTemplate::from_dir(&dir).unwrap();
    let options = DataOptions {
        mask_prompt: true,
        ..DataOptions::default()
    };
    let best = (0..3)
        .map(|_| {
            let start = std::time::Instant::now();
            let ds = Dataset::load_with(
                rows.join("train.jsonl"),
                &tokenizer,
                Some(&template),
                151_645,
                &options,
            )
            .unwrap();
            assert_eq!(ds.len(), 1000);
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min);
    println!("{{\"engine\": \"nuvec\", \"rows\": 1000, \"seconds\": {best}}}");
}
