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
#   - a nightly toolchain with rust-src: `rustup toolchain install nightly
#     --component rust-src`. Pinned in CI to nightly-2026-10-05 (the pages
#     workflow installs it and passes its name through LEAN_NIGHTLY); set
#     LEAN_NIGHTLY to override which nightly `cargo +<toolchain>` below
#     uses, otherwise it defaults to the unpinned `nightly`.
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
#
# Usage: ENGINE_BUILD=<tag> scripts/build_lean_mt.sh
# ENGINE_BUILD is the same `?v=` tag the pages put on their loading URLs;
# the patched worker helper imports `lean.js?v=<tag>` with it, so a worker
# never runs a cached glue file from an older build. MAX_MEMORY (bytes)
# overrides the shared memory's maximum, see below.
set -euo pipefail

: "${ENGINE_BUILD:?set ENGINE_BUILD to the ?v= tag the pages load this build with}"

# Shared wasm memory needs a declared maximum. 1 GiB was too small for
# SmolLM2-1.7B Q4_0: the CPU backend keeps its 0.93 GiB of quantized weights
# resident and allocates a float32 KV cache of 384 KiB per position (0.75 GiB
# at max_ctx 2048), and the load trapped with `unreachable`. Loaded at
# max_ctx 2048 with 8 pool threads, the memory reached 1.96 GiB (2106130432
# bytes) after a 3-turn chat; 2 GiB would leave 40 MiB. 2.5 GiB leaves about
# 0.54 GiB for longer contexts and more pool threads, and stays well under
# wasm32's 4 GiB, since a shared memory's maximum may be reserved up front.
MAX_MEMORY="${MAX_MEMORY:-2684354560}"

# Compiled-in source paths (panic locations, std and registry crates) would
# carry the home directory; map it to a neutral prefix.
REMAP="--remap-path-prefix=$HOME=/home"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="crates/lean/pkg-mt"

echo "==> Building lean (wasm-mt feature, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base $REMAP" \
  cargo +"${LEAN_NIGHTLY:-nightly}" build -p lean --lib \
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
# output name (`lean.js`, not `t0_wasm.js`). Every URL in the chain
# (lean.js -> workerHelpers.js -> the worker's own script -> lean.js)
# carries `?v=$ENGINE_BUILD`, so the workers load the same lean.js module
# the page loaded and nothing is served from an older build's cache.
HELPER=$(find "$OUT_DIR/snippets" -name workerHelpers.js 2>/dev/null | head -1)
if [ -z "$HELPER" ]; then
    echo "error: workerHelpers.js not found under $OUT_DIR/snippets" >&2
    exit 1
fi
sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../lean.js?v=$ENGINE_BUILD')#" \
    -e "s#new URL('\./workerHelpers\.js', import\.meta\.url)#new URL('./workerHelpers.js?v=$ENGINE_BUILD', import.meta.url)#" \
    "$HELPER"
sed -i.bak "s#\(from '\./snippets/[^']*/workerHelpers\.js\)'#\1?v=$ENGINE_BUILD'#" "$OUT_DIR/lean.js"
rm -f "$HELPER.bak" "$OUT_DIR/lean.js.bak"
for want in "lean.js?v=$ENGINE_BUILD" "workerHelpers.js?v=$ENGINE_BUILD"; do
    grep -qF "$want" "$HELPER" || { echo "error: $HELPER lacks $want" >&2; exit 1; }
done
grep -qF "workerHelpers.js?v=$ENGINE_BUILD'" "$OUT_DIR/lean.js" || { echo "error: lean.js does not import workerHelpers.js?v=$ENGINE_BUILD" >&2; exit 1; }
echo "==> Patched $HELPER and lean.js: imports carry ?v=$ENGINE_BUILD"

if strings "$OUT_DIR/lean_bg.wasm" | grep -q "$HOME"; then
    echo "error: $OUT_DIR/lean_bg.wasm still contains the home directory" >&2
    exit 1
fi

echo "==> Wrote $OUT_DIR"
