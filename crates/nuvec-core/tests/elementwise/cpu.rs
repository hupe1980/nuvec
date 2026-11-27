//! Release profile and focused diagnostics for the MLX CPU corpus.

use std::process::Command;

use super::*;

const LEDGER: &str = include_str!("../fixtures/elementwise-cpu-expectations.txt");
const PROFILE: &str = "Apple M4 Pro / 26B131";

fn parse_ledger(text: &str) -> Result<Differences<'_>, String> {
    let mut rows = BTreeMap::new();
    for line in text.lines().filter(|s| !s.starts_with('#')) {
        let mut parts = line.split_whitespace();
        let name = parts.next().ok_or("missing dtype")?;
        let op = parts.next().ok_or("missing operation")?;
        let mut indices = Vec::new();
        for word in parts {
            let index: usize = word.parse().map_err(|_| "invalid chunk index")?;
            if index >= 16 || indices.last().is_some_and(|&last| last >= index) {
                return Err(format!("invalid or unordered index {index} in {name} {op}"));
            }
            indices.push(index);
        }
        if rows.insert((name, op), indices).is_some() {
            return Err(format!("duplicate {name} {op}"));
        }
    }
    if rows.is_empty() {
        return Err("empty CPU ledger".into());
    }
    Ok(rows)
}

fn validate(actual: &Differences<'_>, expected: &Differences<'_>) -> Result<(), String> {
    if actual.keys().ne(expected.keys()) {
        return Err("fixture and ledger case keys differ".into());
    }
    let mut changes = vec![];
    for (key, observed) in actual {
        if expected[key] != *observed {
            changes.push(format!(
                "{} {} expected {:?}, observed {observed:?}",
                key.0, key.1, expected[key]
            ));
        }
    }
    if changes.is_empty() {
        Ok(())
    } else {
        Err(changes.join("\n"))
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn profile_qualified(profile: &str, required: bool) -> Result<bool, String> {
    if profile == PROFILE {
        return Ok(true);
    }
    if required {
        return Err(format!("unqualified CPU profile: {profile}"));
    }
    eprintln!("skipped CPU release ledger: {profile}. Qualified profile: {PROFILE}");
    Ok(false)
}

#[test]
#[cfg_attr(miri, ignore = "physical host profile and millions of values")]
fn qualified_cpu_ledger() {
    let profile = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        let chip = command_output("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_default();
        let os = command_output("/usr/bin/sw_vers", &["-buildVersion"]).unwrap_or_default();
        format!("{chip} / {os}")
    } else {
        std::env::consts::OS.to_owned()
    };
    let required = std::env::var("NUVEC_REQUIRE_CPU_PROFILE").as_deref() == Ok("1");
    if profile_qualified(&profile, required).unwrap() {
        assert_eq!(
            LEDGER.lines().next(),
            Some("# MLX 0.32.5, Apple M4 Pro, macOS 27.2.1 build 26B131.")
        );
        let expected = parse_ledger(LEDGER).unwrap();
        validate(&compare_fixture("cpu"), &expected).unwrap();
    }
}

/// 117-cpu-parity/SR-001 and 117-cpu-parity/SR-002: an incomplete or misattributed run cannot qualify.
#[test]
fn invalid_or_stale_ledger_fails() {
    let expected = parse_ledger("float16 sigmoid\nfloat16 exp 0 1").unwrap();
    for invalid in [
        "",
        "float16 exp 16",
        "float16 exp 1 1",
        "float16 exp\nfloat16 exp",
    ] {
        assert!(parse_ledger(invalid).is_err(), "{invalid}");
    }
    for stale in [
        "float16 sigmoid",
        "float16 sigmoid 0\nfloat16 exp 0 1",
        "float16 sigmoid\nfloat16 exp 0",
    ] {
        assert!(
            validate(&parse_ledger(stale).unwrap(), &expected).is_err(),
            "{stale}"
        );
    }
    assert!(profile_qualified("other", true).is_err());
    assert!(!profile_qualified("other", false).unwrap());
    assert!(profile_qualified(PROFILE, true).unwrap());
}

fn fixture_words(field: &str) -> Vec<u16> {
    if field == "-" {
        return vec![];
    }
    field
        .split(',')
        .map(|s| u16::from_str_radix(s, 16).unwrap())
        .collect()
}

#[test]
fn half_activations_on_views() {
    use nuvec_core::Step;
    const FIXTURE: &str = include_str!("../fixtures/cpu-activations.txt");
    with_stream(Device::cpu(), || {
        let mut lines = FIXTURE.lines();
        assert_eq!(
            lines.next(),
            Some(
                "# MLX CPU 0.32.5: dtype op layout size start step left-bits right-bits expected-bits"
            )
        );
        let mut seen = std::collections::BTreeSet::new();
        for line in lines {
            let fields: [&str; 9] = line
                .split_whitespace()
                .collect::<Vec<_>>()
                .try_into()
                .expect("layout fixture");
            let [dtype, op, layout, size, start, step, left, right, expected] = fields;
            assert!(seen.insert((dtype, op, layout, size)), "duplicate {line}");
            let dtype = dtype_from_name(dtype);
            let size: i32 = size.parse().unwrap();
            let start: i32 = start.parse().unwrap();
            let step: i32 = step.parse().unwrap();
            let backing = |field| {
                let data = fixture_words(field);
                Array::from_slice(&data, [data.len() as i32]).view(dtype)
            };
            let sliced = |field| {
                let view = backing(field).i((start..start + size * step).step(step));
                match layout {
                    "scalar" => view.reshape([0i32; 0]),
                    "transpose" => view.reshape([3, 3]).t(),
                    "row-broadcast" => view.reshape([2, 9]),
                    _ => view,
                }
            };
            let x = sliced(left);
            let y = match layout {
                "broadcast" => backing(right).i(1),
                "row-broadcast" => backing(right).i(..9).reshape([1, 9]),
                _ => sliced(right),
            };
            let out = if op == "sigmoid" {
                x.sigmoid()
            } else {
                x.logaddexp(&y)
            };
            assert_eq!(out.shape(), x.shape(), "{line}");
            assert_eq!(
                canonical_words(&out, dtype),
                fixture_words(expected)
                    .into_iter()
                    .map(u32::from)
                    .collect::<Vec<_>>(),
                "{dtype:?} {op} {layout} size {size}"
            );
        }
        assert_eq!(seen.len(), 40, "incomplete CPU layout fixture");
    });
}

/// scripts/cpu_activation_fixtures.py runs this against the full raw output from the oracle.
/// Keep the original array shape because a scalarized reproducer can take a different SIMD route.
#[test]
#[ignore = "requires an explicitly selected offline oracle"]
fn selected_case_witness() {
    let dtype = dtype_from_name(&std::env::var("NUVEC_WITNESS_DTYPE").expect("dtype"));
    let op = std::env::var("NUVEC_WITNESS_OP").expect("op");
    let reference = std::env::var_os("NUVEC_WITNESS_REF").expect("reference");
    with_stream(Device::cpu(), || {
        let (u, a, b) = inputs(dtype);
        let (out, left) = if op_is_binary(&op) {
            (binary(&op, &a, &b), a)
        } else {
            (unary(&op, &u), u)
        };
        let width = dtype.size();
        let expected_bytes = out.size().checked_mul(width).unwrap();
        assert_eq!(
            std::fs::metadata(&reference).unwrap().len(),
            expected_bytes as u64,
            "wrong reference length"
        );
        let bytes = std::fs::read(reference).unwrap();
        let actual = canonical_words(&out, dtype);
        let expected = bytes.chunks_exact(width).map(|part| {
            if width == 2 {
                u32::from(u16::from_le_bytes(part.try_into().unwrap()))
            } else {
                u32::from_le_bytes(part.try_into().unwrap())
            }
        });
        if let Some((index, (want, got))) =
            expected.zip(actual).enumerate().find(|(_, (e, a))| e != a)
        {
            let operand = |a: &Array| {
                let word = a.i(index as i32);
                if width == 2 {
                    u32::from(word.view(DType::U16).item::<u16>().unwrap())
                } else {
                    word.view(DType::U32).item::<u32>().unwrap()
                }
            };
            let right = if op_is_binary(&op) {
                format!(" right={:#x}", operand(&b))
            } else {
                String::new()
            };
            println!(
                "cpu {dtype:?} {op} index={index} left={:#x}{right} expected={want:#x} observed={got:#x}",
                operand(&left)
            );
        } else {
            println!("cpu {dtype:?} {op}: {} results match", out.size());
        }
    })
}

/// A long evaluation uses cached half-precision transcendentals. Short, lane-aligned
/// evaluations take the direct formulas, including the final scalar tail.
#[test]
fn cached_activations_match_direct_evaluation() {
    with_stream(Device::cpu(), || {
        for dtype in [DType::F16, DType::BF16] {
            let x = values(519, 0).astype(dtype);
            let y = values(519, 521).astype(dtype);
            for op in ["sigmoid", "logaddexp"] {
                let run = |x: &Array, y: &Array| match op {
                    "sigmoid" => x.sigmoid(),
                    _ => x.logaddexp(y),
                };
                let actual = canonical_words(&run(&x, &y), dtype);
                let mut direct = Vec::new();
                for start in (0..519).step_by(8) {
                    let chunk = start..(start + 8).min(519);
                    direct.extend(canonical_words(
                        &run(&x.i(chunk.clone()), &y.i(chunk)),
                        dtype,
                    ));
                }
                assert_eq!(actual, direct, "{dtype:?} {op}");
            }
        }
    });
}
