//! Compiles `src/metal/*.metal` into `nuvec.metallib` with Apple's Metal toolchain.
//!
//! Selects one of three modes and reports it to the crate as `cfg(nuvec_metallib)`:
//!
//! - `precompiled`: the toolchain works (`xcrun metal --version` succeeds), so
//!   `$OUT_DIR/nuvec.metallib` is built. The `.metal` files compile in parallel.
//! - `source`: precompilation is disabled (`NUVEC_METALLIB=off`) or the toolchain is
//!   missing. Shader sources are embedded for runtime compilation, and one warning is printed.
//! - `unavailable`: the target is not macOS. No tools run.
//!
//! Under docs.rs (`DOCS_RS` set) no tool runs either: the mode is `source` with no shaders.
//!
//! `NUVEC_REQUIRE_METALLIB=1` turns every fallback into a build error (used by CI). Any
//! `NUVEC_METALLIB` value other than `off` is ignored with a warning.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;

const OPT_OUT: &str = "NUVEC_METALLIB";
const REQUIRE: &str = "NUVEC_REQUIRE_METALLIB";
const INSTALL_HINT: &str = "install it with `xcodebuild -downloadComponent MetalToolchain`";

/// Deployment floor: macOS 15 (Metal 3.2).
const METAL_STD: &str = "-std=metal3.2";
const MACOS_MIN: &str = "-mmacosx-version-min=15.0";
/// IEEE-conformant math, as MLX compiles its kernels.
const PRECISE: &str = "-fno-fast-math";
/// No implicit fused multiply-add: element-wise results round like the CPU backend.
const NO_CONTRACT: &str = "-ffp-contract=off";
// `CONTRACTED`: shared with the runtime compiler (`nuvec_kernels::contracted`).
include!("src/contracted.rs");

fn main() {
    println!(
        "cargo::rustc-check-cfg=cfg(nuvec_metallib, values(\"precompiled\", \"source\", \"unavailable\"))"
    );
    // A directory is rescanned recursively, so header edits trigger a rebuild too.
    println!("cargo::rerun-if-changed=src/metal");
    println!("cargo::rerun-if-changed=src/contracted.rs");
    println!("cargo::rerun-if-env-changed={OPT_OUT}");
    println!("cargo::rerun-if-env-changed={REQUIRE}");
    println!("cargo::rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo::rerun-if-env-changed=DOCS_RS");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let src_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
            .join("src/metal");
    // docs.rs has no Metal toolchain (and its sandbox has no `xcrun`).
    if env::var_os("DOCS_RS").is_some() {
        println!("cargo::rustc-cfg=nuvec_metallib=\"source\"");
        write_sources(&out_dir, &src_dir, &[]);
        return;
    }
    let files = shader_files(&src_dir);
    let require = env::var(REQUIRE).is_ok_and(|v| v == "1");

    let mut embedded: &[PathBuf] = &[];
    let mode = if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        "unavailable"
    } else if let Some(reason) = fallback_reason(require) {
        println!("cargo::warning={reason}; Metal shaders will be compiled at runtime");
        embedded = &files;
        "source"
    } else {
        precompile(&files, &out_dir);
        "precompiled"
    };

    println!("cargo::rustc-cfg=nuvec_metallib=\"{mode}\"");
    write_sources(&out_dir, &src_dir, embedded);
}

/// Why precompilation cannot run, or `None` if it can. Panics if `require` forbids a fallback.
fn fallback_reason(require: bool) -> Option<String> {
    let opt_out = env::var(OPT_OUT).unwrap_or_default();
    if !matches!(opt_out.as_str(), "" | "off") {
        println!(
            "cargo::warning=ignoring unknown {OPT_OUT}={opt_out:?} (only `off` is recognised)"
        );
    }
    let reason = if opt_out == "off" {
        format!("precompilation disabled by {OPT_OUT}=off")
    } else if !compiler_works() {
        format!("the Metal toolchain (`xcrun metal --version`) is not usable; {INSTALL_HINT}")
    } else if !tool_exists("metallib") {
        format!("the Metal toolchain (`metallib`) was not found; {INSTALL_HINT}")
    } else {
        return None;
    };
    if require {
        panic!("{REQUIRE}=1 but {reason}");
    }
    Some(reason)
}

/// Whether the Metal compiler runs. `xcrun -f metal` is not enough: the `metal` shim exists
/// even when the Metal Toolchain component is not downloaded, and then fails when invoked.
fn compiler_works() -> bool {
    xcrun(&["metal", "--version"]).is_ok_and(|o| o.status.success())
}

fn tool_exists(tool: &str) -> bool {
    xcrun(&["-f", tool]).is_ok_and(|o| o.status.success())
}

fn xcrun(args: &[&str]) -> std::io::Result<Output> {
    Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .args(args)
        .output()
}

/// Compiles every `.metal` file to AIR, in parallel, and links `nuvec.metallib`.
fn precompile(files: &[PathBuf], out_dir: &Path) {
    let debug = env::var("PROFILE").as_deref() == Ok("debug");
    let compile = |src: &PathBuf| -> PathBuf {
        let air = out_dir
            .join(src.file_stem().expect("shader file has a name"))
            .with_extension("air");
        let mut args = vec![METAL_STD, MACOS_MIN, PRECISE];
        if !CONTRACTED.iter().any(|f| src.ends_with(f)) {
            args.push(NO_CONTRACT);
        }
        if debug {
            args.extend(["-gline-tables-only", "-frecord-sources"]);
        }
        let (src_s, air_s) = (path_str(src), path_str(&air));
        args.extend(["-c", src_s, "-o", air_s]);
        run("metal", &args);
        air
    };
    // One thread per file; `run` panics on failure and the panic is re-raised by `join`.
    let airs: Vec<PathBuf> = thread::scope(|scope| {
        let handles: Vec<_> = files
            .iter()
            .filter(|f| f.extension().is_some_and(|e| e == "metal"))
            .map(|src| scope.spawn(move || compile(src)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });

    let lib = out_dir.join("nuvec.metallib");
    let mut args: Vec<&str> = airs.iter().map(|p| path_str(p)).collect();
    args.extend(["-o", path_str(&lib)]);
    run("metallib", &args);
}

/// Runs an `xcrun` tool. On failure, forwards its output verbatim and fails the build.
fn run(tool: &str, args: &[&str]) -> Output {
    let mut full = vec![tool];
    full.extend_from_slice(args);
    let output = xcrun(&full).unwrap_or_else(|e| panic!("failed to run `xcrun {tool}`: {e}"));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        eprint!("{}", String::from_utf8_lossy(&output.stdout));
        panic!("`xcrun {tool}` failed ({})", output.status);
    }
    output
}

/// All shader sources and headers under `dir`, sorted for reproducible output.
fn shader_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).unwrap_or_else(|e| panic!("reading {}: {e}", d.display())) {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "metal" || e == "h") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn write_sources(out_dir: &Path, src_dir: &Path, files: &[PathBuf]) {
    let mut code = String::from("&[");
    for file in files {
        let rel = file.strip_prefix(src_dir).expect("file is under src/metal");
        write!(
            code,
            "({:?}, include_str!({:?})), ",
            path_str(rel),
            path_str(file)
        )
        .expect("writing to a String");
    }
    code.push(']');
    fs::write(out_dir.join("sources.rs"), code).expect("writing sources.rs");
}

fn path_str(path: &Path) -> &str {
    path.to_str()
        .unwrap_or_else(|| panic!("non-UTF-8 path: {}", path.display()))
}
