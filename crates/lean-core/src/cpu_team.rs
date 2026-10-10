//! One parallel region per decode step (and per prefill) for the CPU
//! threads backend.
//!
//! A decode step makes about 170 small matvecs (7 per layer plus the head).
//! Run as separate rayon `par_iter` calls, each one hands work to a pool
//! whose threads have gone to sleep since the previous matvec (rayon parks
//! an idle thread after a few rounds of looking for work, and on wasm32
//! `thread::yield_now` is a no-op, so that takes microseconds), and waking
//! a parked thread is an `Atomics.wait`/`notify` round trip in the browser.
//! The caller, outside the pool, also blocks on a latch for every call.
//!
//! `with_team(f)` instead enters the pool once (`rayon::broadcast`): pool
//! thread 0 runs `f` (the whole step) and every other pool thread spins on
//! an epoch counter. Inside `f`, `team_for(n, body)` (used by both the
//! one-row and the multi-row `linear` in cpu.rs) publishes a job of `n`
//! items; all threads, the leader included, claim items with a CAS on one
//! 64-bit word holding the job's epoch, its item count and the next item,
//! so a thread still looking at an older job can never claim an item of a
//! newer one or read one job's count with another's index, and the leader
//! spins until all `n` are done. A follower that finds no new job for
//! `SPIN_LIMIT` polls parks on a condvar (the leader notifies only when one
//! is parked), so a long serial stretch, or more pool threads than free
//! cores, does not leave threads burning a core that the leader needs. Each
//! item is computed by exactly the code the serial path runs, so results
//! are bit-identical; only the thread that computes an item changes.
//!
//! The thread count is the pool's, a capability read (`initThreadPool(n)`
//! on wasm, the core count natively); nothing is measured or tuned.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// Polls of the epoch word before a follower parks (no `yield_now` on
/// wasm32: each poll is one atomic load).
const SPIN_LIMIT: u32 = 1 << 18;

/// Item count and item index fields of the claim word (24 bits each).
const ITEM_MASK: u64 = (1 << 24) - 1;

type Body<'a> = &'a (dyn Fn(usize) + Sync);

struct Team {
    /// The current job: epoch (low 16 bits) << 48 | item count << 24 |
    /// next unclaimed item, all read and advanced together.
    claim: AtomicU64,
    epoch: AtomicU32,
    done: AtomicUsize,
    /// Address of the leader's `Body` for the current job (a thin pointer
    /// to the fat reference on the leader's stack). Read only after a
    /// successful claim, while the leader is still waiting on that job.
    body: AtomicUsize,
    exit: AtomicBool,
    /// Followers parked (or about to park) on `wake`.
    sleepers: AtomicU32,
    park: Mutex<()>,
    wake: Condvar,
}

thread_local! {
    static TEAM: Cell<*const Team> = const { Cell::new(std::ptr::null()) };
}

impl Team {
    /// Claims and runs items of job `epoch` until none are left.
    fn work(&self, epoch: u32) {
        loop {
            let v = self.claim.load(Ordering::Acquire);
            if (v >> 48) as u16 != epoch as u16 {
                return;
            }
            let (n, i) = ((v >> 24) & ITEM_MASK, v & ITEM_MASK);
            if i >= n {
                return;
            }
            if self.claim.compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire).is_err() {
                continue;
            }
            // SAFETY: item `i` of job `epoch` is claimed and not yet done,
            // so the leader is still inside `run` for this job and its
            // `Body` (whose address it published before the claim word) is
            // alive.
            let body: Body = unsafe { *(self.body.load(Ordering::Relaxed) as *const Body) };
            body(i as usize);
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    fn follow(&self) {
        let mut seen = 0u32;
        let mut idle = 0u32;
        while !self.exit.load(Ordering::SeqCst) {
            let e = self.epoch.load(Ordering::SeqCst);
            if e != seen {
                seen = e;
                idle = 0;
                self.work(e);
                continue;
            }
            idle += 1;
            if idle < SPIN_LIMIT {
                std::hint::spin_loop();
                continue;
            }
            // Park. `sleepers` is raised before the epoch and exit flag are
            // read again under the lock, and the leader reads `sleepers`
            // after publishing, so either this thread sees the new value or
            // the leader sees a sleeper and notifies under the same lock.
            let mut guard = self.park.lock().unwrap();
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            while self.epoch.load(Ordering::SeqCst) == seen && !self.exit.load(Ordering::SeqCst) {
                guard = self.wake.wait(guard).unwrap();
            }
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
            drop(guard);
            idle = 0;
        }
    }

    /// Wakes parked followers after a new epoch or the exit flag.
    fn notify(&self) {
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = self.park.lock().unwrap();
            self.wake.notify_all();
        }
    }

    fn run(&self, n: usize, body: Body) {
        assert!(n as u64 <= ITEM_MASK, "team job of {n} items exceeds the claim word");
        let e = self.epoch.load(Ordering::Relaxed) + 1;
        let body_ref: Body = body;
        self.body.store(&body_ref as *const Body as usize, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        self.claim.store((u64::from(e as u16) << 48) | ((n as u64) << 24), Ordering::Release);
        self.epoch.store(e, Ordering::SeqCst);
        self.notify();
        self.work(e);
        while self.done.load(Ordering::Acquire) < n {
            std::hint::spin_loop();
        }
    }
}

/// Runs `f` on pool thread 0 with the other pool threads spinning, ready
/// for `team_for` (see the module doc). Runs `f` directly when the pool
/// has one thread or a team is already active on this thread.
pub fn with_team<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_num_threads() <= 1 || !TEAM.with(Cell::get).is_null() {
        return f();
    }
    let team = Team {
        claim: AtomicU64::new(0),
        epoch: AtomicU32::new(0),
        done: AtomicUsize::new(0),
        body: AtomicUsize::new(0),
        exit: AtomicBool::new(false),
        sleepers: AtomicU32::new(0),
        park: Mutex::new(()),
        wake: Condvar::new(),
    };
    let f = Mutex::new(Some(f));
    let result = Mutex::new(None);
    rayon::broadcast(|ctx| {
        if ctx.index() == 0 {
            let f = f.lock().unwrap().take().expect("leader runs once");
            TEAM.with(|t| t.set(&team));
            let r = f();
            TEAM.with(|t| t.set(std::ptr::null()));
            team.exit.store(true, Ordering::SeqCst);
            team.notify();
            *result.lock().unwrap() = Some(r);
        } else {
            team.follow();
        }
    });
    result.into_inner().unwrap().expect("leader stored a result")
}

/// Runs `body(i)` for every `i < n` across the active team and returns
/// true, or returns false (running nothing) when no team is active on this
/// thread.
pub fn team_for(n: usize, body: &(dyn Fn(usize) + Sync)) -> bool {
    let team = TEAM.with(Cell::get);
    if team.is_null() {
        return false;
    }
    if n > 0 {
        // SAFETY: TEAM is set only by `with_team`'s leader for the duration
        // of `f`, while `team` lives on `with_team`'s stack.
        unsafe { &*team }.run(n, body);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Many back-to-back jobs of varying sizes, with serial gaps long
    /// enough for followers to park: every item of every job runs exactly
    /// once, and `team_for` returns only when its job is done.
    #[test]
    fn every_item_runs_once_per_job() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        let counts: Vec<AtomicUsize> = (0..512).map(|_| AtomicUsize::new(0)).collect();
        pool.install(|| {
            with_team(|| {
                for job in 0..3000usize {
                    let n = 1 + (job * 7919) % 500;
                    for c in &counts[..n] {
                        c.store(0, Ordering::Relaxed);
                    }
                    assert!(team_for(n, &|i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    }));
                    assert!(counts[..n].iter().all(|c| c.load(Ordering::Relaxed) == 1), "job {job}");
                    if job % 500 == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            })
        });
        assert!(!team_for(1, &|_| {}), "no team outside with_team");
    }
}
