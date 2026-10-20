//! Compiles `src/metal/*.metal` into `nuvec.metallib` with Apple's Metal toolchain.
//!
//! Selects one of three modes and reports it to the crate as `cfg(nuvec_metallib)`:
//!
//! - `precompiled`: the toolchain is available, so `$OUT_DIR/nuvec.metallib` is built.
//! - `source`: precompilation is disabled (`NUVEC_METALLIB=off`) or the toolchain is
//!   missing. Shader sources are embedded for runtime compilation, and one warning is printed.
//! - `unavailable`: the target is not macOS. No tools run.
//!
//! `NUVEC_REQUIRE_METALLIB=1` turns every fallback into a build error (used by CI).

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let src_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
            .join("src/metal");
    let files = shader_files(&src_dir);
    let require = env::var(REQUIRE).is_ok_and(|v| v == "1");

    let mut names: Vec<String> = Vec::new();
    let mut embedded: &[PathBuf] = &[];
    let mode = if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        "unavailable"
    } else if let Some(reason) = fallback_reason(require) {
        println!("cargo::warning={reason}; Metal shaders will be compiled at runtime");
        embedded = &files;
        "source"
    } else {
        names = precompile(&files, &out_dir);
        "precompiled"
    };

    println!("cargo::rustc-cfg=nuvec_metallib=\"{mode}\"");
    write_kernel_names(&out_dir, &names);
    write_sources(&out_dir, &src_dir, embedded);
}

/// Why precompilation cannot run, or `None` if it can. Panics if `require` forbids a fallback.
fn fallback_reason(require: bool) -> Option<String> {
    let reason = if env::var(OPT_OUT).is_ok_and(|v| v == "off") {
        format!("precompilation disabled by {OPT_OUT}=off")
    } else {
        let tool = ["metal", "metallib", "metal-nm"]
            .into_iter()
            .find(|t| !tool_exists(t))?;
        format!("the Metal toolchain (`{tool}`) was not found; {INSTALL_HINT}")
    };
    if require {
        panic!("{REQUIRE}=1 but {reason}");
    }
    Some(reason)
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

/// Compiles every `.metal` file to AIR, links `nuvec.metallib`, and returns its kernel names.
fn precompile(files: &[PathBuf], out_dir: &Path) -> Vec<String> {
    let debug = env::var("PROFILE").as_deref() == Ok("debug");
    let mut airs = Vec::new();
    for src in files
        .iter()
        .filter(|f| f.extension().is_some_and(|e| e == "metal"))
    {
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
        airs.push(air);
    }

    let lib = out_dir.join("nuvec.metallib");
    let mut args: Vec<&str> = airs.iter().map(|p| path_str(p)).collect();
    args.extend(["-o", path_str(&lib)]);
    run("metallib", &args);

    // `metal-nm` lists exported functions as `<addr> T <name>`.
    let nm = run("metal-nm", &[path_str(&lib)]);
    let mut names: Vec<String> = String::from_utf8_lossy(&nm.stdout)
        .lines()
        .filter_map(|l| match l.split_whitespace().collect::<Vec<_>>()[..] {
            [_, "T", name] => Some(name.to_owned()),
            _ => None,
        })
        .collect();
    names.sort();
    names
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

fn write_kernel_names(out_dir: &Path, names: &[String]) {
    let mut code = String::from("&[");
    for name in names {
        write!(code, "{name:?}, ").expect("writing to a String");
    }
    code.push(']');
    fs::write(out_dir.join("kernel_names.rs"), code).expect("writing kernel_names.rs");
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
