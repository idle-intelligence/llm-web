//! Per-component wall-clock profiling of a forward pass, gated by the
//! `LLM_PROFILE` environment variable (unset = every entry point below is a
//! no-op, so the instrumented code paths cost nothing in production).
//!
//! Why it exists: `llm-life`'s variant-B forward went superlinear in T
//! (docs/runs/2026-09-18-forward-scaling.md) and the arithmetic said it
//! should not be, so the first thing needed was a breakdown by component
//! rather than another guess. Burn's wgpu backend dispatches asynchronously,
//! so every measurement here brackets a `Backend::sync` — the numbers are
//! GPU-completion times, not submission times, and the sync itself perturbs
//! the pipeline. Read the table as "where the work is", not as a budget that
//! must sum to the uninstrumented wall clock.
//!
//! WASM: `Instant` is unavailable on `wasm32-unknown-unknown`, so the whole
//! module compiles to no-ops there.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use burn::prelude::Backend;
    use std::cell::RefCell;
    use std::time::Instant;

    thread_local! {
        static ENABLED: bool = std::env::var("LLM_PROFILE").is_ok_and(|v| v != "0");
        /// label -> (total seconds, call count), insertion-ordered.
        static ENTRIES: RefCell<Vec<(&'static str, f64, u64)>> = const { RefCell::new(Vec::new()) };
    }

    pub fn enabled() -> bool {
        ENABLED.with(|e| *e)
    }

    /// Run `f`, and when profiling is on, bracket it with GPU syncs and
    /// accumulate the elapsed time under `label`.
    pub fn scope<T>(label: &'static str, device: &WgpuDevice, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let _ = <Wgpu as Backend>::sync(device);
        let t0 = Instant::now();
        let out = f();
        let _ = <Wgpu as Backend>::sync(device);
        record(label, t0.elapsed().as_secs_f64());
        out
    }

    pub fn record(label: &'static str, secs: f64) {
        ENTRIES.with(|e| {
            let mut e = e.borrow_mut();
            match e.iter_mut().find(|(l, _, _)| *l == label) {
                Some(slot) => {
                    slot.1 += secs;
                    slot.2 += 1;
                }
                None => e.push((label, secs, 1)),
            }
        });
    }

    pub fn reset() {
        ENTRIES.with(|e| e.borrow_mut().clear());
    }

    /// `(label, total seconds, call count)` in first-seen order.
    pub fn take() -> Vec<(&'static str, f64, u64)> {
        ENTRIES.with(|e| std::mem::take(&mut *e.borrow_mut()))
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use burn::backend::wgpu::WgpuDevice;

    pub fn enabled() -> bool {
        false
    }
    pub fn scope<T>(_label: &'static str, _device: &WgpuDevice, f: impl FnOnce() -> T) -> T {
        f()
    }
    pub fn record(_label: &'static str, _secs: f64) {}
    pub fn reset() {}
    pub fn take() -> Vec<(&'static str, f64, u64)> {
        Vec::new()
    }
}

pub use imp::{enabled, record, reset, scope, take};
