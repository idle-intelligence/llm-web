#!/usr/bin/env bash
# Build the single-thread web variant of the lean engine (WebGPU and the
# one-thread CPU backend, SIMD128) into crates/lean/pkg. The threaded CPU
# backend has its own script, scripts/build_lean_mt.sh.
#
# Requires wasm-pack.
#
# Usage: ENGINE_BUILD=<tag> scripts/build_lean.sh
# ENGINE_BUILD is the `?v=` tag the pages put on their loading URLs. This
# build does not write it anywhere; it is required so a rebuild always comes
# with a tag bump on every loading URL, otherwise browsers keep running the
# cached module.
set -euo pipefail

: "${ENGINE_BUILD:?set ENGINE_BUILD to the ?v= tag the pages load this build with}"

# Compiled-in source paths (panic locations, std and registry crates) would
# carry the home directory; map it to a neutral prefix. RUSTFLAGS replaces
# .cargo/config.toml's rustflags, so +simd128 is repeated here.
export RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix=$HOME=/home"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="crates/lean/pkg"
WASM="$OUT_DIR/lean_bg.wasm"

echo "==> Building lean (web feature, single thread) for ENGINE_BUILD=$ENGINE_BUILD"
wasm-pack build crates/lean --target web --release --no-default-features --features web

# Local paths and the user name must not survive into the binary.
LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
if [ -n "$LEAKS$USER_HITS" ]; then
    echo "error: $WASM contains local paths or the user name:" >&2
    printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
    exit 1
fi

echo "==> Wrote $OUT_DIR ($(shasum -a 256 "$WASM" | cut -c1-16))"
