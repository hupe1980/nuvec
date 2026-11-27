//! Library capacity, packed import, rollback, and exact decode using deterministic small Qwen3s.
use nuvec::{Array, DType, Device, Result, Stream, ops};
use nuvec_lm::{
    Batcher, BatcherOptions, Config, GenerateOptions, KvQuant, Model, PagedCache, decode_batch,
};
use nuvec_nn::Module;

fn model(d: usize, gqa: usize, dtype: DType) -> Model {
    let mut c = Config::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_f32/config.json"
    ))
    .unwrap();
    c.head_dim = d;
    c.num_attention_heads = gqa;
    c.num_key_value_heads = 1;
    c.num_hidden_layers = 1;
    let mut m = Model::new(&c, &|_| None).unwrap();
    let params = m
        .parameters()
        .into_iter()
        .enumerate()
        .map(|(i, (name, p))| {
            let shape = p.shape();
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let data: Vec<f32> = (0..n)
                .map(|j| {
                    if shape.len() == 1 {
                        1.0
                    } else {
                        let h = (j as u32)
                            .wrapping_mul(0x9e3779b9)
                            .wrapping_add(i as u32)
                            .wrapping_mul(0x85ebca6b);
                        (((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0 * 2.0 - 1.0) * 0.1
                    }
                })
                .collect();
            (name, Array::from_slice(&data, shape).astype(dtype))
        })
        .collect::<Vec<_>>();
    m.load_weights(params).unwrap();
    m
}

fn q() -> KvQuant {
    KvQuant {
        start: 0,
        ..Default::default()
    }
}

#[test]
/// 118-paged-quantized-kv/SR-001 and SR-003: invalid capacities, mismatched imports and rollback.
fn packed_capacity_format_validation_and_reuse() -> Result<()> {
    nuvec::set_default_stream(Stream::default_for(Device::cpu()));
    let m = model(64, 2, DType::BF16);
    let small = model(16, 2, DType::F32);
    assert!(PagedCache::new_quantized(&small, 2, 4, 1).is_err());
    assert!(PagedCache::new_quantized(&m, usize::MAX, 4, 1).is_err());
    let mut batch = Batcher::new_quantized(
        &m,
        BatcherOptions {
            blocks: Some(12),
            ..Default::default()
        },
    )?;
    assert!(
        batch
            .submit(
                vec![1],
                GenerateOptions {
                    kv_quant: Some(q()),
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(batch.submit(vec![1], Default::default())?, 0);

    let mut pool = PagedCache::new_quantized(&m, 8, 4, 3)?;
    let dense = PagedCache::new(&m, 8, 4, 3)?;
    let mut wrong_format = m.new_cache();
    wrong_format[0].to_quantized(KvQuant { bits: 4, ..q() })?;
    assert!(pool.insert(&wrong_format).is_err());
    assert_eq!(pool.free_blocks(), 8);
    let mut wrong_dims = model(64, 2, DType::BF16);
    wrong_dims.config.head_dim = 128;
    assert!(PagedCache::new_quantized(&wrong_dims, 8, 4, 3).is_err());
    assert_eq!(
        pool.kv_capacity_bytes() * 128,
        dense.kv_capacity_bytes() * 68
    );
    let budget = 1024 * 1024;
    let packed_budget =
        PagedCache::new_quantized(&m, budget / (pool.kv_capacity_bytes() / 8), 4, 3)?;
    let dense_budget = PagedCache::new(&m, budget / (dense.kv_capacity_bytes() / 8), 4, 3)?;
    assert!(
        packed_budget.kv_capacity_bytes() <= budget && dense_budget.kv_capacity_bytes() <= budget
    );
    assert!(packed_budget.free_blocks() * 10 >= dense_budget.free_blocks() * 18);
    let mut kv = m.new_cache();
    let _ = m.forward(&Array::from_slice(&[1u32, 2, 3, 4, 5], [1, 5]), &mut kv);
    assert!(pool.insert(&kv).is_err());
    kv[0].to_quantized(q())?;
    let survivor = pool.insert(&kv)?;
    let snapshot = |p: &PagedCache| -> Result<Vec<_>> {
        p.state()
            .into_iter()
            .map(|a| ops::view(&a.contiguous(), DType::U8).to_vec::<u8>())
            .collect()
    };
    assert_eq!(
        pool.state().iter().map(Array::nbytes).sum::<usize>(),
        pool.kv_capacity_bytes()
    );
    let free = pool.free_blocks();
    // A failed multi-row import must return the first row's partial reservation.
    let mut too_big = m.new_cache();
    too_big[0].to_quantized(q())?;
    let ids = ops::zeros([2, 16], DType::U32);
    let _ = m.forward(&ids, &mut too_big);
    let before = snapshot(&pool)?;
    assert!(pool.insert_many(&too_big).is_err());
    assert_eq!(pool.free_blocks(), free);
    assert_eq!(snapshot(&pool)?, before);
    // Keep one sequence alive as a succession of short requests reuses its
    // neighbour's blocks. Occasionally advance the survivor across a page.
    let mut failed_model = Model::new(&m.config, &|_| None)?;
    failed_model.load_weights_partial(
        m.parameters()
            .into_iter()
            .filter(|(name, _)| name == "model.embed_tokens.weight")
            .collect::<Vec<_>>(),
    )?;
    assert!(decode_batch(&small, &[42], &[survivor], &mut pool).is_err());
    assert_eq!(pool.len(survivor), Some(5));
    let mut other = m.new_cache();
    other[0].to_quantized(q())?;
    let _ = m
        .forward(&Array::from_slice(&[10u32, 11, 12, 13], [1, 4]), &mut other)
        .check()?;
    for cycle in 0..100 {
        let new = pool.insert(&other)?;
        let before = snapshot(&pool)?;
        let free_before = pool.free_blocks();
        let length_before = pool.len(survivor);
        assert!(decode_batch(&failed_model, &[22, 41], &[survivor, new], &mut pool).is_err());
        assert_eq!(pool.free_blocks(), free_before);
        assert_eq!(pool.len(survivor), length_before);
        assert_eq!(pool.len(new), Some(4));
        assert_eq!(snapshot(&pool)?, before);
        let mut short = other.clone();
        let advance = cycle % 12 == 0;
        let (tokens, ids) = if advance {
            (vec![22, 41], vec![survivor, new])
        } else {
            (vec![41], vec![new])
        };
        let got = nuvec::with_row_exact(|| decode_batch(&m, &tokens, &ids, &mut pool))?;
        for (r, &token) in tokens.iter().enumerate() {
            let history = if ids[r] == survivor {
                &mut kv
            } else {
                &mut short
            };
            let expected =
                nuvec::with_row_exact(|| m.forward(&Array::from_slice(&[token], [1, 1]), history))
                    .reshape([-1])
                    .astype(DType::F32);
            assert_eq!(
                got.i(r as i32).astype(DType::F32).to_vec::<f32>()?,
                expected.to_vec::<f32>()?
            );
        }
        pool.remove(new)?;
        assert_eq!(
            pool.free_blocks(),
            8 - pool.len(survivor).unwrap().div_ceil(4)
        );
    }
    pool.remove(survivor)?;
    assert_eq!(pool.free_blocks(), 8);
    Ok(())
}

#[test]
/// 118-paged-quantized-kv/SR-002: distinct sequences see exactly their own compressed history.
fn packed_exact_steps_match_solo_histories() -> Result<()> {
    for device in [Device::cpu(), Device::gpu(0)] {
        if device == Device::gpu(0) && !nuvec::metal::is_available() {
            continue;
        }
        nuvec::set_default_stream(Stream::default_for(device));
        for (d, g, dtype, bs) in [
            (64, 1, DType::F32, 1),
            (128, 2, DType::F16, 2),
            (64, 4, DType::BF16, 4),
            (128, 8, DType::F32, 8),
        ] {
            let m = model(d, g, dtype);
            let mut pages = PagedCache::new_quantized(&m, bs * 24, 4, bs)?;
            let mut histories = Vec::new();
            let mut ids = Vec::new();
            let mut next = Vec::new();
            for i in 0..bs {
                let mut hist = m.new_cache();
                for h in &mut hist {
                    h.to_quantized(q())?;
                }
                if i > 0 {
                    let prompt: Vec<u32> = (0..i + 2).map(|n| n as u32 + 3 + i as u32).collect();
                    let _ = m
                        .forward(&Array::from(prompt).reshape([1, -1]), &mut hist)
                        .check()?;
                }
                ids.push(pages.insert(&hist)?);
                histories.push(hist);
                next.push(i as u32 + 10);
            }
            for _ in 0..32 {
                let got = nuvec::with_row_exact(|| decode_batch(&m, &next, &ids, &mut pages))?
                    .argmax(-1)
                    .to_vec::<u32>()?;
                for i in 0..bs {
                    let want = nuvec::with_row_exact(|| {
                        m.forward(&Array::from_slice(&[next[i]], [1, 1]), &mut histories[i])
                    })
                    .argmax(-1)
                    .item::<u32>()?;
                    assert_eq!(
                        got[i], want,
                        "{device:?} {dtype:?} head {d}, batch {bs}, row {i}"
                    );
                }
                next = got;
            }
        }
    }
    Ok(())
}

#[test]
fn packed_batch_preemption_and_cancellation() -> Result<()> {
    use nuvec_lm::{Event, FinishReason};
    nuvec::set_default_stream(Stream::default_for(Device::cpu()));
    let mut m = model(64, 2, DType::F32);
    m.config.eos_token_ids.clear();
    let prompts = [vec![2, 3, 5], vec![17, 13, 11], vec![31, 19, 23]];
    let count = 17;
    let mut batch = Batcher::new_quantized(
        &m,
        BatcherOptions {
            blocks: Some(7),
            block_size: 4,
            max_batch: 3,
            ..Default::default()
        },
    )?;
    for (i, prompt) in prompts.iter().enumerate() {
        assert_eq!(
            batch.submit(
                prompt.clone(),
                GenerateOptions {
                    max_tokens: count,
                    row_exact: true,
                    ..Default::default()
                }
            )?,
            i
        );
    }
    let mut outputs = [vec![], vec![], vec![]];
    let mut reasons = [None, None, None];
    for step in 0..100 {
        if !batch.is_busy() {
            break;
        }
        if step == 1 {
            assert!(batch.cancel(2));
        }
        for event in batch.step()? {
            match event {
                Event::Token { id, token } => outputs[id].push(token),
                Event::Finished { id, reason } => assert!(reasons[id].replace(reason).is_none()),
            }
        }
    }
    assert!(!batch.is_busy());
    assert!(batch.preemptions() > 0);
    assert_eq!(
        reasons,
        [
            Some(FinishReason::Length),
            Some(FinishReason::Length),
            Some(FinishReason::Cancelled)
        ]
    );
    assert_eq!(outputs[2].len(), 1);
    for i in 0..2 {
        let mut history = m.new_cache();
        history[0].to_quantized(q())?;
        nuvec::with_row_exact(|| -> Result<()> {
            let prompt = &prompts[i];
            let (last, head) = prompt.split_last().unwrap();
            let _ = m
                .forward(&Array::from_slice(head, [1, head.len()]), &mut history)
                .check()?;
            let mut token = *last;
            assert_eq!(outputs[i].len(), count);
            for &got in &outputs[i] {
                token = m
                    .forward(&Array::from_slice(&[token], [1, 1]), &mut history)
                    .argmax(-1)
                    .item::<u32>()?;
                assert_eq!(got, token);
            }
            Ok(())
        })?;
    }
    Ok(())
}
