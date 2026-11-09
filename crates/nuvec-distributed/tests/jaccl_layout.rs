//! The RDMA bindings' struct sizes and field offsets against the SDK's
//! `infiniband/verbs.h`: a C probe compiled with the system compiler prints every value the
//! bindings rely on. Skipped without `cc` or the SDK.

#![cfg(all(feature = "jaccl", target_os = "macos"))]

use std::process::Command;

#[test]
fn bindings_match_the_sdk_header() {
    let Ok(sdk) = Command::new("xcrun").arg("--show-sdk-path").output() else {
        eprintln!("skipped: no xcrun");
        return;
    };
    let sdk = String::from_utf8(sdk.stdout).unwrap().trim().to_string();
    if !std::path::Path::new(&sdk)
        .join("usr/include/infiniband/verbs.h")
        .exists()
    {
        eprintln!("skipped: the SDK has no infiniband/verbs.h");
        return;
    }
    let layout = nuvec_distributed::jaccl::verbs_layout();
    let mut probe = String::from(
        "#include <stdio.h>\n#include <stddef.h>\n#include <infiniband/verbs.h>\nint main(void) {\n",
    );
    for (expr, _) in &layout {
        probe.push_str(&format!("  printf(\"%zu\\n\", (size_t){expr});\n"));
    }
    probe.push_str("  return 0;\n}\n");
    let dir = std::env::temp_dir().join(format!("nuvec-verbs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, exe) = (dir.join("probe.c"), dir.join("probe"));
    std::fs::write(&src, probe).unwrap();
    let built = Command::new("cc")
        .args(["-isysroot", &sdk, "-o"])
        .arg(&exe)
        .arg(&src)
        .output();
    let Ok(built) = built else {
        eprintln!("skipped: no cc");
        return;
    };
    assert!(
        built.status.success(),
        "the probe does not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let out = Command::new(&exe).output().unwrap();
    let values: Vec<usize> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    let wrong: Vec<String> = layout
        .iter()
        .zip(&values)
        .filter(|((_, rust), c)| rust != *c)
        .map(|((expr, rust), c)| format!("{expr}: Rust {rust}, C {c}"))
        .collect();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(values.len(), layout.len());
    assert!(wrong.is_empty(), "layouts differ:\n{}", wrong.join("\n"));
}

#[test]
fn the_library_loads_and_lists_this_macs_devices() {
    // RDMA is off on a Mac until `rdma_ctl enable` in recoveryOS: the library still loads
    // (macOS 26.2+) and lists no devices.
    match nuvec_distributed::jaccl::devices() {
        Ok(devices) => {
            for d in &devices {
                eprintln!("{} {:?} {:?}", d.name, d.port_state, d.ipv4);
            }
            assert_eq!(
                nuvec_distributed::jaccl::is_available(),
                !devices.is_empty()
            );
        }
        Err(e) => {
            assert!(e.to_string().contains("not available"), "{e}");
            assert!(!nuvec_distributed::jaccl::is_available());
        }
    }
}
