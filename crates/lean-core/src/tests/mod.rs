//! Tests that reach into private modules (GGUF parsing, dequantization).
//! Tests on the public API are in `tests/`.

mod dequant_reference;
mod embed_head_sliced;
mod q6k_reference_dequant;
mod stop_tokens;
