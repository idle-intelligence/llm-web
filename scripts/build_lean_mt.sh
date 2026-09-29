#!/usr/bin/env bash
# Build the threaded-wasm variant of the lean engine's CPU rung into
# crates/lean/pkg-mt. Separate from the default single-thread `crates/lean/pkg`
# build (crates/lean/Cargo.toml's `web` feature): the atomics/bulk-memory/
# mutable-globals target features and the `-Z build-std` nightly flag only
# apply to this invocation, never to .cargo/config.toml (which would break
# the single-thread pkg build's +simd128-only rustflags).
#
# Same recipe as t0-web's tools/build-mt.sh (idle-intelligence/t0-web PR #8,
# "Threads spike for the CPU build: measured, not adopted") - the project's own prior
# work, reused here, not reinvented: t0 built a burn-ndarray/rayon threaded
# wasm module the same way (nightly build-std + wasm-bindgen-rayon +
# explicit atomics/shared-memory linker flags), because wasm-pack cannot
# pass `-Z build-std` through to cargo (it lands after cargo's own `--`,
# where cargo treats it as a rustc flag and errors).
#
# Requires:
#   - nightly toolchain with rust-src: `rustup toolchain install nightly
#     --component rust-src`
#   - wasm-bindgen-cli matching the wasm-bindgen version in Cargo.lock
#     exactly (`cargo install wasm-bindgen-cli --version <that version>
#     --locked`)
#
# Output: crates/lean/pkg-mt (lean.js, lean_bg.wasm, snippets/... with
# wasm-bindgen-rayon's workerHelpers.js). The page must:
#   1. `await init(wasmUrl)` (wasm module init)
#   2. `await initThreadPool(navigator.hardwareConcurrency)` (spins up the
#      rayon worker pool - lib.rs's `wasm-mt`-gated re-export)
#   3. then call LeanEngineCpu as usual
# and must be served cross-origin isolated (COOP/COEP) - see
# scripts/serve_coi.py - for SharedArrayBuffer + atomics to work at all.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="crates/lean/pkg-mt"

echo "==> Building lean (wasm-mt feature, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
  cargo +nightly build -p lean --lib \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features --features wasm-mt \
    -Z build-std=panic_abort,std

WASM_IN="target/wasm32-unknown-unknown/release/lean.wasm"
if [ ! -f "$WASM_IN" ]; then
    echo "error: expected build output not found at $WASM_IN" >&2
    exit 1
fi

echo "==> Running wasm-bindgen"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name lean "$WASM_IN"

# wasm-bindgen-rayon's generated workerHelpers.js does a bundler-relative
# `import('../../..')` to reach the main module, which doesn't resolve for
# a plain browser `import()` of a bare directory (`--target web`, no
# bundler) - same fix t0-web's build-mt.sh applies, adapted to this crate's
# output name (`lean.js`, not `t0_wasm.js`).
HELPER=$(find "$OUT_DIR/snippets" -name workerHelpers.js 2>/dev/null | head -1)
if [ -n "$HELPER" ]; then
    sed -i.bak "s#await import('\.\./\.\./\.\.')#await import('../../../lean.js')#" "$HELPER"
    rm -f "$HELPER.bak"
    echo "==> Patched $HELPER for --target web bare-directory import"
fi

echo "==> Wrote $OUT_DIR"
