#!/usr/bin/env bash
# Builds both lean web variants and assembles the deployed site into _site/.
#
#   crates/lean/pkg    WebGPU + single-thread CPU (SIMD128)
#   crates/lean/pkg-mt CPU threads (needs a nightly toolchain with rust-src;
#                       wasm-pack cannot pass `-Z build-std` through to
#                       cargo, so this calls cargo directly and runs
#                       wasm-bindgen by hand - same reasoning as t0-web's
#                       tools/build-mt.sh, idle-intelligence/t0-web PR #8)
#
# then rewrites the `?v=` build tag to ENGINE_BUILD on every loading URL
# (web/lean-chat-worker.js, web/device/device_worker.js, web/device/index.html)
# and checks neither built wasm carries a local build path.
#
# Requires wasm-pack, wasm-bindgen-cli (version matching Cargo.lock's
# wasm-bindgen exactly: `cargo install wasm-bindgen-cli --version <ver>
# --locked`), and a nightly toolchain with rust-src installed.
#
# Usage: ENGINE_BUILD=<tag> scripts/build.sh
# ENGINE_BUILD defaults to "dev" for local builds; CI passes the commit sha.
# It is required on every real deploy - a rebuild with no tag bump keeps
# browsers running the cached module.
# LEAN_NIGHTLY selects the nightly for the pkg-mt build, default
# nightly-2026-09-20 (the toolchain pinned in the workflow).
set -euo pipefail

ENGINE_BUILD="${ENGINE_BUILD:-dev}"
LEAN_NIGHTLY="${LEAN_NIGHTLY:-nightly-2026-09-20}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> nightly toolchain: $(cargo +"$LEAN_NIGHTLY" --version)"

# --- crates/lean/pkg: WebGPU + single-thread CPU ---
# Compiled-in source paths (panic locations, std and registry crates) would
# carry the home directory; map it to a neutral prefix. RUSTFLAGS replaces
# .cargo/config.toml's rustflags, so +simd128 is repeated here.
echo "==> Building lean (web feature, single thread) for ENGINE_BUILD=$ENGINE_BUILD"
RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix=$HOME=/home" \
  wasm-pack build crates/lean --target web --release --no-default-features --features web

ST_WASM="crates/lean/pkg/lean_bg.wasm"

# --- crates/lean/pkg-mt: CPU threads ---
# Separate from the single-thread pkg build above: the atomics/bulk-memory/
# mutable-globals target features and the `-Z build-std` nightly flag must
# never land in .cargo/config.toml (that would break the single-thread
# pkg build's +simd128-only rustflags).
#
# Shared wasm memory needs a declared maximum. 1 GiB was too small for
# SmolLM2-1.7B Q4_0: the CPU backend keeps its 0.93 GiB of quantized weights
# resident and allocates a float32 KV cache of 384 KiB per position (0.75 GiB
# at max_ctx 2048), and the load trapped with `unreachable`. Loaded at
# max_ctx 2048 with 8 pool threads, the memory reached 1.96 GiB (2106130432
# bytes) after a 3-turn chat; 2 GiB would leave 40 MiB. 2.5 GiB leaves about
# 0.54 GiB for longer contexts and more pool threads, and stays well under
# wasm32's 4 GiB, since a shared memory's maximum may be reserved up front.
MAX_MEMORY="${MAX_MEMORY:-2684354560}"

echo "==> Building lean (wasm-mt feature, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base \
--remap-path-prefix=$HOME=/home" \
  cargo +"${LEAN_NIGHTLY}" build -p lean --lib \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features --features wasm-mt \
    -Z build-std=panic_abort,std

WASM_IN="target/wasm32-unknown-unknown/release/lean.wasm"
if [ ! -f "$WASM_IN" ]; then
    echo "error: expected build output not found at $WASM_IN" >&2
    exit 1
fi

MT_OUT="crates/lean/pkg-mt"
echo "==> Running wasm-bindgen"
rm -rf "$MT_OUT"
mkdir -p "$MT_OUT"
wasm-bindgen --target web --out-dir "$MT_OUT" --out-name lean "$WASM_IN"

# wasm-bindgen-rayon's generated workerHelpers.js does a bundler-relative
# `import('../../..')` to reach the main module, which doesn't resolve for
# a plain browser `import()` of a bare directory (`--target web`, no
# bundler). Every URL in the chain (lean.js -> workerHelpers.js -> the
# worker's own script -> lean.js) carries `?v=$ENGINE_BUILD`, so the
# workers load the same lean.js module the page loaded and nothing is
# served from an older build's cache.
HELPER=$(find "$MT_OUT/snippets" -name workerHelpers.js 2>/dev/null | head -1)
if [ -z "$HELPER" ]; then
    echo "error: workerHelpers.js not found under $MT_OUT/snippets" >&2
    exit 1
fi
sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../lean.js?v=$ENGINE_BUILD')#" \
    -e "s#new URL('\./workerHelpers\.js', import\.meta\.url)#new URL('./workerHelpers.js?v=$ENGINE_BUILD', import.meta.url)#" \
    "$HELPER"
sed -i.bak "s#\(from '\./snippets/[^']*/workerHelpers\.js\)'#\1?v=$ENGINE_BUILD'#" "$MT_OUT/lean.js"
rm -f "$HELPER.bak" "$MT_OUT/lean.js.bak"
for want in "lean.js?v=$ENGINE_BUILD" "workerHelpers.js?v=$ENGINE_BUILD"; do
    grep -qF "$want" "$HELPER" || { echo "error: $HELPER lacks $want" >&2; exit 1; }
done
grep -qF "workerHelpers.js?v=$ENGINE_BUILD'" "$MT_OUT/lean.js" || { echo "error: lean.js does not import workerHelpers.js?v=$ENGINE_BUILD" >&2; exit 1; }
echo "==> Patched $HELPER and lean.js: imports carry ?v=$ENGINE_BUILD"

MT_WASM="$MT_OUT/lean_bg.wasm"

# --- Local-path / user-name leak check on both wasm outputs ---
for WASM in "$ST_WASM" "$MT_WASM"; do
    LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $WASM contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
done
echo "==> $ST_WASM and $MT_WASM: no local paths"

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site/web/device _site/web/lib _site/web/lean _site/pkg
cp web/index.html web/lean-chat-worker.js web/apple-touch-icon.png web/favicon.ico _site/web/
cp web/lib/model-cache.js _site/web/lib/
cp -R web/device/. _site/web/device/
cp -R crates/lean/pkg _site/web/lean/pkg
cp -R crates/lean/pkg-mt _site/web/lean/pkg-mt
cp -R pkg/wllama _site/pkg/wllama

# --- Rewrite the ?v= build tag to ENGINE_BUILD on every loading URL ---
echo "==> Rewriting ENGINE_BUILD tag to $ENGINE_BUILD"
sed -i "s/const ENGINE_BUILD = \"[^\"]*\"/const ENGINE_BUILD = \"$ENGINE_BUILD\"/" \
  _site/web/lean-chat-worker.js _site/web/device/device_worker.js
sed -i "s/const ENGINE_BUILD = \"[^\"]*\";/const ENGINE_BUILD = \"$ENGINE_BUILD\";/" \
  _site/web/device/index.html

COUNT="$(grep -rEo 'ENGINE_BUILD = "[^"]*"' _site/web/lean-chat-worker.js _site/web/device/device_worker.js _site/web/device/index.html | grep -Fc "\"$ENGINE_BUILD\"")"
if [ "$COUNT" -ne 3 ]; then
    echo "error: expected 3 ENGINE_BUILD assignments rewritten to $ENGINE_BUILD, found $COUNT" >&2
    exit 1
fi

echo "==> Wrote _site"
