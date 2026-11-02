//! Checkpoint footprints (what a load maps, copies and briefly holds), the refusal before
//! mapping, misaligned and filtered loads (bit-identical to the file), and asynchronous
//! saves. The memory checks read the shared GPU device's
//! statistics, so they run one at a time (`MEMORY`).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use nuvec_core::{Array, DType, MappedFile, ops};
use nuvec_io::{
    load_safetensors, load_safetensors_filtered, safetensors_footprint,
    safetensors_footprint_filtered, save_safetensors, save_safetensors_async,
};

static MEMORY: Mutex<()> = Mutex::new(());

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nuvec-st-{}-{name}", std::process::id()))
}

/// An MLX-style file (an unpadded header: the data section starts misaligned) with one
/// float32 tensor of `n` elements.
fn misaligned(path: &Path, n: usize) {
    let h = format!(
        "{{\"x\":{{\"dtype\":\"F32\",\"shape\":[{n}],\"data_offsets\":[0,{}]}}}}",
        n * 4
    );
    let mut h = h.into_bytes();
    if (8 + h.len()).is_multiple_of(4) {
        h.push(b' ');
    }
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(&h);
    bytes.extend((0..n).flat_map(|i| (i as f32).to_le_bytes()));
    std::fs::write(path, bytes).unwrap();
}

fn device() -> Option<&'static nuvec_gpu::Device> {
    let d = nuvec_gpu::Device::shared();
    if d.is_none() {
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
    }
    d
}

#[test]
fn footprints_match_what_a_load_maps_and_copies() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = device() else { return };
    let dir = temp("dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Aligned (written by nuvec, as by Hugging Face): mapped whole, header included.
    let aligned = dir.join("a.safetensors");
    let w = ops::arange(0, 300_000, 1).astype(DType::F32);
    save_safetensors(&aligned, &[("w", &w)], &[]).unwrap();
    let len = std::fs::metadata(&aligned).unwrap().len();
    // Misaligned (as MLX writes): the tensors are read, the file is not mapped.
    let mlx = dir.join("b.safetensors");
    misaligned(&mlx, 400_000);

    let before = dev.memory();
    let fa = safetensors_footprint(&aligned).unwrap();
    let fb = safetensors_footprint(&mlx).unwrap();
    let fd = safetensors_footprint(&dir).unwrap();
    // Headers only: nothing mapped or allocated.
    let after = dev.memory();
    assert_eq!((after.mapped, after.active), (before.mapped, before.active));
    assert_eq!(
        (fa.mapped, fa.copied, fa.transient),
        (
            MappedFile::mapping_size(len),
            0,
            MappedFile::mapping_size(len)
        )
    );
    assert!(fa.mapped >= len);
    let copied = MappedFile::copy_size(400_000 * 4);
    assert_eq!((fb.mapped, fb.copied), (0, copied));
    assert_eq!(fb.transient, copied);
    assert_eq!(fd.mapped, fa.mapped);
    assert_eq!(fd.copied, fb.copied);
    assert_eq!(fd.transient, fa.mapped + fb.copied);

    // Loading: the deltas equal the footprints, and the peak rises by the transient
    // (within a page).
    for (path, fp) in [(&aligned, fa), (&mlx, fb)] {
        let start = dev.memory();
        dev.reset_peak();
        let st = load_safetensors(path).unwrap();
        let m = dev.memory();
        assert_eq!((m.mapped - start.mapped) as u64, fp.mapped, "{path:?}");
        assert_eq!((m.active - start.active) as u64, fp.copied, "{path:?}");
        let rise = (m.peak_with_mapped - start.active_with_mapped) as u64;
        assert!(
            rise <= fp.transient,
            "{path:?}: rise {rise} > {}",
            fp.transient
        );
        assert!(
            rise + (16 << 10) >= fp.transient,
            "{path:?}: rise {rise} vs {}",
            fp.transient
        );
        drop(st);
    }

    // A limit below the transient refuses before mapping or reading.
    let start = dev.memory();
    let limit = start.active_with_mapped + fb.transient as usize / 2;
    let previous = dev.set_memory_limit(limit);
    let err = load_safetensors(&mlx).unwrap_err();
    dev.set_memory_limit(previous);
    assert!(err.to_string().contains("b.safetensors"), "{err}");
    assert!(err.to_string().contains("out of memory"), "{err}");
    let end = dev.memory();
    assert_eq!(
        (end.mapped, end.active),
        (start.mapped, start.active),
        "nothing held"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn broken_files_are_errors_naming_the_file() {
    let truncated = temp("truncated.safetensors");
    let w = ops::arange(0, 100, 1).astype(DType::F32);
    save_safetensors(&truncated, &[("w", &w)], &[]).unwrap();
    let bytes = std::fs::read(&truncated).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() - 7]).unwrap();
    let text = temp("notes.safetensors");
    std::fs::write(&text, "these are not tensors, just some words").unwrap();
    for path in [&truncated, &text] {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let err = safetensors_footprint(path).unwrap_err().to_string();
        assert!(err.contains(&name), "{err}");
        let err = load_safetensors(path).unwrap_err().to_string();
        assert!(err.contains(&name), "{err}");
        std::fs::remove_file(path).unwrap();
    }
    assert!(safetensors_footprint(temp("missing.safetensors")).is_err());
}

#[test]
fn async_saves_equal_sync_saves() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    // Arrays still being computed (a chain of 100 dependent operations) when the save
    // starts.
    let mut x = ops::arange(0, 4096, 1).astype(DType::F32).reshape([64, 64]);
    for _ in 0..100 {
        x = (x.exp() * 0.5).sqrt() * 0.999 + 0.001;
    }
    let half = x.astype(DType::BF16);
    let (a, b) = (temp("async.safetensors"), temp("sync.safetensors"));
    let tensors = [("x", &x), ("half", &half)];
    let meta = [("step", "7")];
    let handle = save_safetensors_async(&a, &tensors, &meta);
    handle.wait().unwrap();
    save_safetensors(&b, &tensors, &meta).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let mut tmp = a.clone().into_os_string();
    tmp.push(".tmp");
    assert!(!PathBuf::from(tmp).exists(), "no temporary file left");
    // Dropping the handle still completes the write.
    let c = temp("dropped.safetensors");
    drop(save_safetensors_async(&c, &tensors, &meta));
    assert_eq!(std::fs::read(&c).unwrap(), std::fs::read(&b).unwrap());
    // Write errors come back from `wait`.
    let missing = temp("no-such-dir").join("x.safetensors");
    let err = save_safetensors_async(&missing, &tensors, &meta).wait();
    assert!(err.is_err());
    for p in [a, b, c] {
        std::fs::remove_file(p).unwrap();
    }
    let _: &Array = &x;
}

/// A safetensors file with a header of `pad` extra bytes (unpadded, as MLX writes, when
/// that makes the data start misaligned) holding `tensors` in order; returns each tensor's
/// bytes.
fn write_raw(path: &Path, pad: usize, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut entries = Vec::new();
    let mut at = 0;
    for (name, dtype, shape, bytes) in tensors {
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{at},{}]}}",
            dims.join(","),
            at + bytes.len()
        ));
        at += bytes.len();
    }
    let mut h = format!("{{{}}}", entries.join(",")).into_bytes();
    h.extend(std::iter::repeat_n(b' ', pad));
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(&h);
    for (.., bytes) in tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
}

/// The little-endian bytes of a tensor, as nuvec reads them back.
fn bytes_of(a: &Array) -> Vec<u8> {
    let st = temp("bytes-of.safetensors");
    save_safetensors(&st, &[("a", a)], &[]).unwrap();
    let file = std::fs::read(&st).unwrap();
    let start = 8 + u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    file[start..].to_vec()
}

/// Every dtype, odd element counts and a header length at every offset modulo 8 (so most
/// tensors start misaligned): every tensor loads bit-identical to its bytes in the file,
/// aligned in memory, also when a filter leaves gaps.
#[test]
fn misaligned_files_load_bit_identical() {
    // Deterministic pseudo-random bytes.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect()
    };
    // Bool and float bytes must be valid values: bools 0/1; floats any bits are fine.
    let dtypes: [(&str, usize); 13] = [
        ("U8", 1),
        ("BOOL", 1),
        ("I8", 1),
        ("U16", 2),
        ("I16", 2),
        ("F16", 2),
        ("BF16", 2),
        ("U32", 4),
        ("I32", 4),
        ("F32", 4),
        ("U64", 8),
        ("I64", 8),
        ("F64", 8),
    ];
    let mut tensors = Vec::new();
    for (i, (dtype, size)) in dtypes.iter().enumerate() {
        let n = 2 * i + 3; // odd counts: odd-length u8 tensors misalign what follows
        let mut bytes = random(n * size);
        if *dtype == "BOOL" {
            bytes.iter_mut().for_each(|b| *b &= 1);
        }
        tensors.push((format!("t{i:02}"), *dtype, vec![n], bytes));
    }
    // A large tensor read in several chunks by several threads.
    tensors.push(("big".into(), "U32", vec![9_000_001], random(36_000_004)));
    let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
        .collect();
    for pad in 0..8 {
        let path = temp(&format!("raw-{pad}.safetensors"));
        write_raw(&path, pad, &tensors);
        let all = load_safetensors(&path).unwrap();
        assert_eq!(all.tensors.len(), tensors.len());
        for (name, _, shape, bytes) in &tensors {
            let a = &all.tensors[*name];
            assert_eq!(
                a.shape(),
                shape.iter().map(|&d| d as i32).collect::<Vec<_>>()
            );
            assert_eq!(&bytes_of(a), bytes, "{name}, pad {pad}");
        }
        // Every other tensor: the kept ones are identical, the others absent.
        let keep = |name: &str| name.ends_with(['0', '2', '4', '6', '8']) || name == "big";
        let some = load_safetensors_filtered(&path, keep).unwrap();
        let mut kept: Vec<&str> = tensors.iter().map(|t| t.0).filter(|n| keep(n)).collect();
        kept.sort_unstable();
        assert_eq!(
            some.tensors.keys().map(String::as_str).collect::<Vec<_>>(),
            kept
        );
        for name in kept {
            assert_eq!(bytes_of(&some.tensors[name]), bytes_of(&all.tensors[name]));
        }
        assert!(
            load_safetensors_filtered(&path, |_| false)
                .unwrap()
                .tensors
                .is_empty()
        );
        std::fs::remove_file(&path).unwrap();
    }
}

/// Filtered loads of misaligned files read (and hold) only the kept tensors.
#[test]
fn filtered_loads_hold_only_kept_tensors() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = device() else { return };
    let path = temp("filtered.safetensors");
    let n = 1 << 20;
    let tensor = |v: u8| vec![v; n * 4];
    write_raw(
        &path,
        1,
        &[
            ("model.layers.0.w", "F32", vec![n], tensor(1)),
            ("model.visual.w", "F32", vec![n], tensor(2)),
            ("model.layers.1.w", "F32", vec![n], tensor(3)),
        ],
    );
    let start = dev.memory();
    let st = load_safetensors_filtered(&path, |k| !k.starts_with("model.visual")).unwrap();
    let m = dev.memory();
    assert_eq!(m.mapped, start.mapped, "not mapped");
    let held = (m.active - start.active) as u64;
    assert!(
        held >= 2 * 4 * n as u64 && held < 3 * 4 * n as u64,
        "held {held}"
    );
    assert_eq!(
        st.tensors["model.layers.1.w"].to_vec::<f32>().unwrap()[0],
        f32::from_le_bytes([3; 4])
    );
    drop(st);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn filtered_footprints_count_only_kept_tensors_of_misaligned_files() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = device() else { return };
    // An MLX-style file with a kept and a dropped float32 tensor.
    let path = temp("filtered");
    let (kept, dropped) = (300_000usize, 500_000usize);
    let h = format!(
        "{{\"keep\":{{\"dtype\":\"F32\",\"shape\":[{kept}],\"data_offsets\":[0,{}]}},\
         \"drop\":{{\"dtype\":\"F32\",\"shape\":[{dropped}],\"data_offsets\":[{},{}]}}}}",
        kept * 4,
        kept * 4,
        (kept + dropped) * 4
    );
    let mut h = h.into_bytes();
    if (8 + h.len()).is_multiple_of(4) {
        h.push(b' ');
    }
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(&h);
    bytes.extend((0..kept + dropped).flat_map(|i| (i as f32).to_le_bytes()));
    std::fs::write(&path, bytes).unwrap();

    let keep = |name: &str| name == "keep";
    let all = safetensors_footprint(&path).unwrap();
    let some = safetensors_footprint_filtered(&path, keep).unwrap();
    assert_eq!(
        all.copied,
        MappedFile::copy_size(((kept + dropped) * 4) as u64)
    );
    assert_eq!(
        (some.mapped, some.copied),
        (0, MappedFile::copy_size((kept * 4) as u64))
    );
    assert_eq!(some.transient, some.copied);
    // The filtered load allocates exactly that.
    let start = dev.memory();
    let st = load_safetensors_filtered(&path, keep).unwrap();
    let m = dev.memory();
    assert_eq!((m.active - start.active) as u64, some.copied);
    assert_eq!(st.tensors.len(), 1);
    drop(st);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tensors_of_one_file_share_a_buffer_id() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    // Aligned (mapped) and misaligned (read into one buffer per file).
    let aligned = temp("ids-aligned");
    let (a, b) = (
        ops::arange(0, 64, 1).astype(DType::F32),
        ops::arange(0, 32, 1).astype(DType::I32),
    );
    save_safetensors(&aligned, &[("a", &a), ("b", &b)], &[]).unwrap();
    let mlx = temp("ids-mlx");
    misaligned(&mlx, 1000);
    for path in [&aligned, &mlx] {
        let st = load_safetensors(path).unwrap();
        let ids: Vec<_> = st.tensors.values().map(|t| t.buffer_id()).collect();
        assert!(
            ids.iter().all(|id| id.is_some() && *id == ids[0]),
            "{path:?}: {ids:?}"
        );
    }
    let other = load_safetensors(&aligned).unwrap();
    let st = load_safetensors(&mlx).unwrap();
    assert_ne!(other.tensors["a"].buffer_id(), st.tensors["x"].buffer_id());
    let _ = (std::fs::remove_file(&aligned), std::fs::remove_file(&mlx));
}

/// A filtered load of an aligned file maps only the kept tensors' pages: the mapped bytes
/// equal the footprint, the dropped tensor's pages are not counted, and the kept tensors
/// equal a whole load's.
#[test]
fn filtered_aligned_loads_map_only_the_kept_pages() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = device() else { return };
    let path = temp("filtered-aligned");
    let first = ops::arange(0, 1000, 1).astype(DType::F32);
    let dropped = ops::arange(0, 4 << 20, 1).astype(DType::F32); // 16 MiB
    let last = ops::arange(0, 3000, 1).astype(DType::BF16);
    save_safetensors(
        &path,
        &[
            ("a.first", &first),
            ("b.dropped", &dropped),
            ("c.last", &last),
        ],
        &[],
    )
    .unwrap();
    let keep = |name: &str| name != "b.dropped";
    let whole = safetensors_footprint(&path).unwrap();
    let some = safetensors_footprint_filtered(&path, keep).unwrap();
    assert!(some.mapped < whole.mapped / 8, "{some:?} vs {whole:?}");
    assert_eq!((some.copied, some.transient), (0, some.mapped));
    let before = dev.memory().mapped;
    let st = load_safetensors_filtered(&path, keep).unwrap();
    assert_eq!((dev.memory().mapped - before) as u64, some.mapped);
    let all = load_safetensors(&path).unwrap();
    for name in ["a.first", "c.last"] {
        let (x, y) = (&st.tensors[name], &all.tensors[name]);
        assert_eq!(
            x.astype(DType::F32).to_vec::<f32>().unwrap(),
            y.astype(DType::F32).to_vec::<f32>().unwrap(),
            "{name}"
        );
    }
    assert!(!st.tensors.contains_key("b.dropped"));
    drop((st, all));
    let _ = std::fs::remove_file(&path);
}

/// An MLX-style file with metadata and tensors of odd sizes: every tensor misaligned.
fn misaligned_mixed(path: &Path) -> Vec<(&'static str, DType, Vec<u8>)> {
    let tensors: Vec<(&str, DType, Vec<u8>)> = vec![
        (
            "a.weight",
            DType::F32,
            (0..5 * 4).map(|i| (i * 7) as u8).collect(),
        ),
        (
            "b.scales",
            DType::BF16,
            (0..7 * 2).map(|i| (i * 13 + 1) as u8).collect(),
        ),
        ("c.codes", DType::U8, vec![9, 8, 7]),
        (
            "d.big",
            DType::F32,
            (0..3000 * 4).map(|i| (i % 251) as u8).collect(),
        ),
    ];
    let names = ["F32", "BF16", "U8", "F32"];
    let mut at = 0;
    let mut entries = vec![r#""__metadata__":{"format":"mlx"}"#.to_string()];
    for ((name, dtype, bytes), dname) in tensors.iter().zip(names) {
        let n = bytes.len() / dtype.size();
        entries.push(format!(
            r#""{name}":{{"dtype":"{dname}","shape":[{n}],"data_offsets":[{at},{}]}}"#,
            at + bytes.len()
        ));
        at += bytes.len();
    }
    let mut h = format!("{{{}}}", entries.join(",")).into_bytes();
    if (8 + h.len()).is_multiple_of(4) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(&h);
    for (_, _, b) in &tensors {
        out.extend(b);
    }
    std::fs::write(path, out).unwrap();
    tensors
}

fn raw(a: &Array) -> Vec<u8> {
    a.view(DType::U8).to_vec::<u8>().unwrap()
}

/// `align_safetensors` keeps every tensor and the metadata bit for bit and makes the file
/// mappable; padding tensors never surface.
#[test]
fn aligned_copies_keep_every_tensor_and_map_without_a_copy() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let (src, dst) = (temp("align-src"), temp("align-dst"));
    let tensors = misaligned_mixed(&src);
    assert!(
        safetensors_footprint(&src).unwrap().copied > 0,
        "the source is read, not mapped"
    );
    nuvec_io::align_safetensors(&src, &dst).unwrap();
    let fp = safetensors_footprint(&dst).unwrap();
    assert_eq!((fp.copied, fp.mapped > 0), (0, true), "{fp:?}");
    let (a, b) = (
        load_safetensors(&src).unwrap(),
        load_safetensors(&dst).unwrap(),
    );
    assert_eq!(b.metadata, a.metadata);
    assert_eq!(b.tensors.len(), tensors.len());
    for (name, _, bytes) in &tensors {
        assert_eq!(&raw(&b.tensors[*name]), bytes, "{name}");
    }
    let listed = nuvec_io::safetensors_tensors(&dst).unwrap();
    assert_eq!(listed.len(), tensors.len());
    assert!(!dst.with_extension("safetensors.tmp").exists());
    let _ = (std::fs::remove_file(&src), std::fs::remove_file(&dst));
}

/// The cache writes an aligned copy once, maps it afterwards, rebuilds it when the source
/// changes, and falls back to reading when it cannot write.
#[test]
fn the_aligned_cache_is_written_once_and_rebuilt_on_change() {
    let _one = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let src = temp("cache-src.safetensors");
    let dir = temp("cache-dir");
    let _ = std::fs::remove_dir_all(&dir);
    let tensors = misaligned_mixed(&src);
    let copies = || std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
    let first = nuvec_io::load_safetensors_cached(&src, &dir, |_| true).unwrap();
    assert_eq!(copies(), 1);
    let copy = std::fs::read_dir(&dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert_eq!(safetensors_footprint(&copy).unwrap().copied, 0);
    let again = nuvec_io::load_safetensors_cached(&src, &dir, |n| n != "d.big").unwrap();
    assert_eq!(copies(), 1);
    assert_eq!(again.tensors.len(), tensors.len() - 1);
    for (name, _, bytes) in &tensors {
        assert_eq!(&raw(&first.tensors[*name]), bytes, "{name}");
    }
    // A changed source (another length) gets a new copy.
    drop((first, again));
    std::thread::sleep(std::time::Duration::from_millis(20));
    misaligned(&src, 33);
    let changed = nuvec_io::load_safetensors_cached(&src, &dir, |_| true).unwrap();
    assert_eq!(copies(), 2);
    assert_eq!(changed.tensors["x"].to_vec::<f32>().unwrap()[32], 32.0);
    // An unwritable cache (a file in the way) still loads, by reading.
    let blocked = temp("cache-blocked");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let read = nuvec_io::load_safetensors_cached(&src, blocked.join("sub"), |_| true).unwrap();
    assert_eq!(read.tensors.len(), 1);
    let _ = (
        std::fs::remove_file(&src),
        std::fs::remove_dir_all(&dir),
        std::fs::remove_file(&blocked),
    );
}
