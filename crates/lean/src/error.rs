//! The error type a one-turn `generate` call returns. Most failures are
//! just "something went wrong" (tokenizer, chat-template rendering, the
//! forward pass itself) and carry no structure a caller could act on; the
//! one failure a caller *can* act on - the rendered conversation plus
//! `max_new_tokens` does not fit `max_ctx` - gets its own variant so a
//! client can decide how to trim and retry without parsing a string.

use std::fmt;

#[derive(Debug)]
pub enum LeanError {
    /// The rendered conversation's token count plus `max_new_tokens`
    /// exceeds `max_ctx`. `needed` is that sum; `available` is `max_ctx`.
    /// The engine's own state (KV cache, cached prefix) is untouched - this
    /// is checked before anything is prefilled, so the next call still
    /// works normally.
    ContextOverflow { needed: usize, available: usize },
    /// Anything else (tokenizer, chat-template rendering, the forward
    /// pass).
    Other(anyhow::Error),
}

impl fmt::Display for LeanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeanError::ContextOverflow { needed, available } => {
                write!(f, "context overflow: needed {needed} tokens, {available} available")
            }
            LeanError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LeanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LeanError::ContextOverflow { .. } => None,
            LeanError::Other(e) => e.source(),
        }
    }
}

impl From<anyhow::Error> for LeanError {
    fn from(e: anyhow::Error) -> Self {
        LeanError::Other(e)
    }
}
