# OVERVIEW — llm-wasm crate layout

`llm-web` is about running LLMs in the browser on Burn + wgpu (WebGPU, GGUF
models) — the engine and its WASM bindings are the point. The MCP agent
loop demo (`web/agent/`) and `eval/` are applications built on top of that
engine, not the project itself; Sonos-specific fixtures/prompts live only
under `fixtures/sonos` and `eval/` and are not a dependency of the core
engine or the headless harness.

Skeleton for xLAM-2-3b-fc-r (Qwen2 architecture) in Burn+wgpu, mirroring stt-web's
stt-wasm engine (see docs/ENGINE.md, docs/MODELS.md). `Cargo.toml` (workspace root
and crate) and `src/lib.rs` are frozen for the two phases below — don't touch them;
everything else is fair game.

```
Cargo.toml                    workspace: Burn 0.20, cubecl 0.9 wgpu, wgpu 26, tokenizers 0.22
patches/cubecl-wgpu-0.9.0/    workgroup-size patch, copied from stt-web
scripts/headless/             general headless-Chromium harness: load+run any page/model
                               via window.__llm, verify + benchmark decode tok/s
crates/llm-wasm/
  Cargo.toml                  features: wgpu, native, web
  src/
    lib.rs                    LlmConfig + module declarations (frozen)
    gguf.rs                   [1b] copied from stt-wasm, needs Qwen2 tensor-name rewrite
    model.rs                  [1b] type skeleton (stt-web shapes) -> real Qwen2 arch
    kv.rs                     [1b] stub -> growable/paged KV cache
    sample.rs                 [1b] stub -> temperature/top_p/top_k/rep-penalty sampling
    wgsl/shader_naive.wgsl    [1b] copied from stt-wasm, Q4_0 naive dequant+matmul
    template.rs                [1a] stub -> minijinja chat_template rendering
    tools.rs                  [1a] stub -> MCP tool schema + xLAM tool-call parsing
    agent.rs                  [1a] stub -> agent loop (generate -> parse -> tool -> repeat)
    tokenizer.rs               [1a] stub -> tokenizers wrapper (fancy-regex backend)
    web.rs                    [1b bindings / 1a calls] stub -> wasm-bindgen surface
    bin/llm-agent.rs           native CLI stub (run/eval/gguf-info), required-features=["native"]
  tests/
    template.rs, tools.rs, agent.rs        [1a] placeholders
    q4_matmul.rs, full_forward.rs          [1b] placeholders, cfg(feature = "wgpu")
```

## Ownership

- **Phase 1a** (no GPU/Burn needed): `template.rs`, `tools.rs`, `agent.rs`,
  `tokenizer.rs`, `tests/{template,tools,agent}.rs`, fixtures under `fixtures/`.
- **Phase 1b** (Burn/wgpu/GGUF): `gguf.rs`, `model.rs`, `kv.rs`, `sample.rs`,
  `wgsl/`, `tests/{q4_matmul,full_forward}.rs`, `src/bin/llm-agent.rs` body.
  Owns rewriting `gguf.rs`'s tensor-name table + `model.rs`'s struct shapes
  together for Qwen2 (q/k/v bias, 3-matrix SwiGLU, tied lm_head, no sliding
  window, `blk.N.*` naming) — see file header/module doc comments.

Both phases may touch `LlmModel`/`Q4Attention`/etc. in `model.rs` freely; only
`Cargo.toml` (root + crate) and `lib.rs` are frozen.

## Build commands

```bash
# Native (default features: wgpu + native)
CARGO_BUILD_JOBS=4 cargo build
CARGO_BUILD_JOBS=4 cargo test
CARGO_BUILD_JOBS=4 cargo clippy --all-targets

# WASM (web feature; native feature/clap/CLI excluded)
CARGO_BUILD_JOBS=4 cargo build --target wasm32-unknown-unknown --no-default-features --features web -p llm-wasm
CARGO_BUILD_JOBS=4 cargo clippy --target wasm32-unknown-unknown --no-default-features --features web -p llm-wasm

# wasm-pack (not yet run in this skeleton — do this once bindings exist)
wasm-pack build crates/llm-wasm --target web --no-default-features --features web
```

Always set `CARGO_BUILD_JOBS=4` — this machine has limited memory and a full
`burn`+`cubecl`+`wgpu` build is heavy.
