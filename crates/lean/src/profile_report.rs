//! Opt-in kernel-time aggregation for `LEAN_PROFILE_KERNELS=1` - see
//! `engine.rs`'s module doc comment for the feature-gate rule. Every
//! profiled decode step's per-dispatch GPU timestamps
//! (`Engine::read_profile`) are folded in here via `record_step`, with the
//! per-layer index stripped from each dispatch's call-site label the same
//! way `model.rs::scratch_key` strips it for `Pool` (`"dec_layer5.norm"` ->
//! `"dec_layer.norm"`), so one row covers "this kernel, every layer" summed
//! across the whole run rather than one row per layer per step.
//! `lean_cli.rs` calls `report()` once, after its decode loop, to print the
//! table (stderr, so it never disturbs a case's stdout timing/parity line).
//! Thread-local, not global: this binary is single-threaded for its GPU
//! work, and a thread-local avoids adding a `Mutex` for a diagnostics-only
//! path that is off by default.

use std::cell::RefCell;
use std::collections::BTreeMap;

thread_local! {
    static TOTALS: RefCell<BTreeMap<String, (u64, f64)>> = const { RefCell::new(BTreeMap::new()) };
}

fn strip_layer(label: &str) -> String {
    let (head, rest) = match label.split_once('.') {
        Some((h, r)) => (h, Some(r)),
        None => (label, None),
    };
    let stripped = head.trim_end_matches(|c: char| c.is_ascii_digit());
    match rest {
        Some(r) => format!("{stripped}.{r}"),
        None => stripped.to_string(),
    }
}

/// Folds one decode step's `(label, gpu_ns)` list into the running totals.
pub fn record_step(data: &[(String, f64)]) {
    TOTALS.with(|t| {
        let mut t = t.borrow_mut();
        for (label, ns) in data {
            let key = strip_layer(label);
            let entry = t.entry(key).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += ns;
        }
    });
}

/// Prints the aggregated table and clears it. `steps` is the number of
/// decode steps folded in, used only to report each kernel's average
/// per-step dispatch count as a sanity check against `decode_dispatches_per_step`.
pub fn report(steps: u64) {
    TOTALS.with(|t| {
        let mut t = t.borrow_mut();
        if t.is_empty() {
            return;
        }
        eprintln!("kernel_profile steps={steps}");
        eprintln!("kernel,count,count_per_step,total_us,us_per_call");
        let mut rows: Vec<_> = t.iter().collect();
        rows.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
        for (label, (count, total_ns)) in rows {
            let total_us = total_ns / 1000.0;
            let us_per_call = total_us / *count as f64;
            let count_per_step = *count as f64 / steps as f64;
            eprintln!("{label},{count},{count_per_step:.2},{total_us:.2},{us_per_call:.3}");
        }
        t.clear();
    });
}
