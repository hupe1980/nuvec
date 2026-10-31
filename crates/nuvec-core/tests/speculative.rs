//! On-device acceptance (`speculative::accept_greedy`, `accept_sampled`): equal to a host
//! reference on random cases (dtypes, sliced and transposed logits, drafts of 0–12 tokens,
//! EOS sets of 0–3 ids, u32 and i32 drafts), the sampled rule's output distributed as the
//! target's, seeded results repeated, invalid inputs failed.

use nuvec_core::{Array, DType, Device, Stream, random, set_default_stream, speculative};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The reference: (accepted, next, eos_hit).
fn reference(choices: &[u32], draft: &[u32], eos: &[u32]) -> (usize, u32, bool) {
    let mut accepted = 0;
    let mut hit = false;
    for (d, c) in draft.iter().zip(choices) {
        if d != c {
            break;
        }
        accepted += 1;
        if eos.contains(d) {
            hit = true;
            break;
        }
    }
    let next = choices[accepted];
    (accepted, next, hit || eos.contains(&next))
}

fn devices() -> Vec<Device> {
    let mut d = vec![Device::cpu()];
    if cfg!(target_os = "macos") && !cfg!(miri) {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        if (&Array::from(1.0f32) + 1.0).eval().is_ok() {
            d.push(Device::gpu(0));
        }
    }
    d
}

#[test]
fn greedy_acceptance_equals_the_reference() {
    let mut rng = Rng(0x1234_5678);
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        for case in 0..400 {
            let (n, v, e) = (rng.below(13), 3 + rng.below(30), rng.below(4));
            // Logits with a clear argmax per row.
            let choices: Vec<u32> = (0..=n).map(|_| rng.below(v) as u32).collect();
            let mut values = vec![0f32; (n + 1) * v];
            for (row, &c) in choices.iter().enumerate() {
                for j in 0..v {
                    values[row * v + j] = (rng.below(100) as f32) / 100.0;
                }
                values[row * v + c as usize] = 5.0;
            }
            // Drafts: a prefix of the choices, then something else.
            let agree = rng.below(n + 1);
            let draft: Vec<u32> = (0..n)
                .map(|i| match i < agree {
                    true => choices[i],
                    false => (choices[i] + 1 + rng.below(v - 1) as u32) % v as u32,
                })
                .collect();
            let eos: Vec<u32> = (0..e)
                .map(|_| match rng.below(3) {
                    0 if !draft.is_empty() => draft[rng.below(draft.len())],
                    _ => rng.below(v) as u32,
                })
                .collect();
            let want = reference(&choices, &draft, &eos);
            let dtype = [DType::F32, DType::F16, DType::BF16][case % 3];
            // Layouts: contiguous, a slice of a [1, L, V] array with an offset, transposed.
            let base = Array::from_slice(&values, [n as i32 + 1, v as i32]).astype(dtype);
            let logits = match case % 3 {
                0 => base,
                1 => {
                    let pad = nuvec_core::ops::zeros([2, v as i32], dtype);
                    nuvec_core::ops::concatenate(&[pad, base], 0)
                        .reshape([1, n as i32 + 3, v as i32])
                        .i((0, 2.., ..))
                }
                _ => base.t().contiguous().t(),
            };
            let draft_a = match case % 2 {
                0 => Array::from_slice(&draft, [n as i32]),
                _ => Array::from_slice(&draft, [n as i32]).astype(DType::I32),
            };
            let eos_a = Array::from_slice(&eos, [e as i32]);
            let out = speculative::accept_greedy(&logits, &draft_a, &eos_a);
            let (a, next, hit) = speculative::read(&out).unwrap();
            assert_eq!(
                (a, next, hit),
                want,
                "case {case} on {device:?}: {draft:?} {choices:?} {eos:?}"
            );
        }
    }
}

#[test]
fn a_device_draft_is_accepted_without_reading_it() {
    // The draft is the argmax of another array: lazy, never read on the host.
    let logits = Array::from_slice(&[0.0f32, 9.0, 9.0, 0.0, 0.0, 9.0], [3, 2]);
    let draft = logits.i(0..2).argmax(-1);
    let none = Array::from_slice(&[0u32; 0], [0]);
    let out = speculative::accept_greedy(&logits, &draft, &none);
    assert_eq!(speculative::read(&out).unwrap(), (2, 1, false));
}

#[test]
fn invalid_inputs_fail_with_a_message() {
    let logits = Array::from_slice(&[0.0f32; 6], [3, 2]);
    let none = Array::from_slice(&[0u32; 0], [0]);
    let two = Array::from_slice(&[0u32, 1], [2]);
    let cases = [
        speculative::accept_greedy(&logits, &Array::from_slice(&[0u32], [1]), &none),
        speculative::accept_greedy(&logits, &two.astype(DType::F32), &none),
        speculative::accept_greedy(&logits.astype(DType::I32), &two, &none),
        speculative::accept_sampled(&logits, &two, Some(&logits), &none, &random::key(0)),
    ];
    for (i, c) in cases.iter().enumerate() {
        let err = c.clone().check().unwrap_err().to_string();
        assert!(err.contains("accept_"), "case {i}: {err}");
    }
}

/// The first emitted token (the draft token if accepted, else the residual sample) is
/// distributed as the target's `p`: for a fixed (point-mass) draft and for drafts drawn
/// from `q` with their distribution given. Frequencies over 3,000 keys within 5 standard
/// errors of `p`.
#[test]
fn sampled_acceptance_keeps_the_target_distribution() {
    set_default_stream(Stream::default_for(Device::cpu()));
    let p = [0.5f32, 0.3, 0.15, 0.05];
    let q = [0.1f32, 0.2, 0.3, 0.4];
    let probs = Array::from_slice(&[p, p].concat(), [2, 4]);
    let none = Array::from_slice(&[0u32; 0], [0]);
    let runs = 3000;
    let mut rng = Rng(99);
    for stochastic in [false, true] {
        let mut counts = [0usize; 4];
        for seed in 0..runs {
            // A point mass on token 2, or a draft drawn from q.
            let d = match stochastic {
                false => 2u32,
                true => {
                    let u = (rng.below(1_000_000) as f32) / 1e6;
                    let mut acc = 0.0;
                    q.iter()
                        .position(|&x| {
                            acc += x;
                            u < acc
                        })
                        .unwrap_or(3) as u32
                }
            };
            let draft = Array::from_slice(&[d], [1]);
            let qa = Array::from_slice(&q, [1, 4]);
            let out = speculative::accept_sampled(
                &probs,
                &draft,
                stochastic.then_some(&qa),
                &none,
                &random::key(seed),
            );
            let (a, next, _) = speculative::read(&out).unwrap();
            let first = if a == 1 { d } else { next };
            counts[first as usize] += 1;
        }
        for (t, &c) in counts.iter().enumerate() {
            let f = c as f64 / runs as f64;
            let se = (p[t] as f64 * (1.0 - p[t] as f64) / runs as f64).sqrt();
            assert!(
                (f - p[t] as f64).abs() < 5.0 * se + 1e-3,
                "stochastic {stochastic}: token {t} at {f:.3}, p {}",
                p[t]
            );
        }
    }
    // The same key repeats; a fully accepted draft samples its bonus from the last row.
    let draft = Array::from_slice(&[0u32], [1]);
    let sure = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2, 4]);
    let a = speculative::accept_sampled(&sure, &draft, None, &none, &random::key(5));
    let b = speculative::accept_sampled(&sure, &draft, None, &none, &random::key(5));
    assert_eq!(
        speculative::read(&a).unwrap(),
        speculative::read(&b).unwrap()
    );
    assert_eq!(speculative::read(&a).unwrap(), (1, 3, false));
}

/// No gradient flows through acceptance (integer outputs, as `argmax`), and `vmap` over a
/// leading batch of verification steps equals the steps one by one.
#[test]
fn acceptance_has_no_gradient_and_batches() {
    set_default_stream(Stream::default_for(Device::cpu()));
    let logits = Array::from_slice(&[0.0f32, 9.0, 1.0, 9.0, 0.0, 1.0, 0.0, 1.0, 9.0], [3, 3]);
    let draft = Array::from_slice(&[1u32, 0], [2]);
    let none = Array::from_slice(&[0u32; 0], [0]);
    let g = nuvec_core::grad(|l: &Array| {
        speculative::accept_greedy(l, &draft, &none)
            .astype(DType::F32)
            .sum(..)
    })(&logits)
    .unwrap();
    assert!(g.to_vec::<f32>().unwrap().iter().all(|&x| x == 0.0));
    // Two verification steps at once.
    let other = Array::from_slice(&[9.0f32, 0.0, 1.0, 0.0, 9.0, 1.0, 1.0, 0.0, 9.0], [3, 3]);
    let batch = nuvec_core::ops::stack(&[logits.clone(), other], 0);
    let drafts = nuvec_core::ops::stack(&[draft.clone(), Array::from_slice(&[1u32, 2], [2])], 0);
    let out = nuvec_core::vmap(|(l, d): &(Array, Array)| speculative::accept_greedy(l, d, &none))
        .call(&(batch.clone(), drafts.clone()))
        .unwrap();
    for i in 0..2 {
        let one = speculative::accept_greedy(&batch.i(i), &drafts.i(i), &none);
        assert_eq!(
            out.i(i).to_vec::<u32>().unwrap(),
            one.to_vec::<u32>().unwrap(),
            "step {i}"
        );
    }
}
