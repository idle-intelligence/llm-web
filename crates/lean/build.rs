// Sets LEAN_ENGINE_HASH: a hash of every source file of lean and lean-core
// (Rust and WGSL), of the workspace Cargo.lock (which pins wgpu and its
// shader compiler) and rust-toolchain.toml, and of the target architecture
// and features (a native build and a wasm build run different CPU kernels).
// A prefix KV image records it, so an image written by one build of the
// engine is never restored into another: a kernel change can change the
// floats of the cache.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let roots = [manifest.join("src"), manifest.join("../lean-core/src"), manifest.join("../../Cargo.lock"), manifest.join("../../rust-toolchain.toml")];
    let mut list = Vec::new();
    for r in &roots {
        println!("cargo:rerun-if-changed={}", r.display());
        if r.is_dir() {
            files(r, &mut list);
        } else if r.exists() {
            list.push(r.clone());
        }
    }
    let mut keyed: Vec<(String, PathBuf)> = list.into_iter().map(|p| (p.strip_prefix(&manifest).map(|q| q.to_string_lossy().replace('\\', "/")).unwrap_or_default(), p)).collect();
    keyed.sort();
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64, 0x6c62_272e_07bb_0142u64);
    for (name, path) in &keyed {
        let bytes = std::fs::read(path).unwrap_or_default();
        for part in [name.as_bytes(), &(bytes.len() as u64).to_le_bytes(), &bytes] {
            a = fnv(a, part);
            b = fnv(b, part);
        }
    }
    for var in ["CARGO_CFG_TARGET_ARCH", "CARGO_CFG_TARGET_FEATURE"] {
        let v = std::env::var(var).unwrap_or_default();
        a = fnv(a, v.as_bytes());
        b = fnv(b, v.as_bytes());
    }
    println!("cargo:rustc-env=LEAN_ENGINE_HASH={a:016x}{b:016x}");
}
