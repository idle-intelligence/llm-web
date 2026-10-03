//! One parallel region per decode step for the CPU threads backend.
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
//! an epoch counter. Inside `f`, `team_for(n, body)` publishes a job of `n`
//! items; all threads, the leader included, claim items with a CAS on one
//! 64-bit word (job epoch in the high half, next item in the low half, so a
//! late thread can never claim an item of a newer job), and the leader
//! spins until all `n` are done. No thread sleeps until `f` returns. Each
//! item is computed by exactly the code the serial path runs, so results
//! are bit-identical; only the thread that computes an item changes.
//!
//! The thread count is the pool's, a capability read (`initThreadPool(n)`
//! on wasm, the core count natively); nothing is measured or tuned.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

type Body<'a> = &'a (dyn Fn(usize) + Sync);

struct Team {
    /// `(epoch << 32) | next unclaimed item` of the current job.
    claim: AtomicU64,
    epoch: AtomicU32,
    n: AtomicUsize,
    done: AtomicUsize,
    /// Address of the leader's `Body` for the current job (a thin pointer
    /// to the fat reference on the leader's stack). Read only after a
    /// successful claim, while the leader is still waiting on that job.
    body: AtomicUsize,
    exit: AtomicBool,
}

thread_local! {
    static TEAM: Cell<*const Team> = const { Cell::new(std::ptr::null()) };
}

impl Team {
    /// Claims and runs items of job `epoch` until none are left.
    fn work(&self, epoch: u32) {
        loop {
            let v = self.claim.load(Ordering::Acquire);
            if (v >> 32) as u32 != epoch {
                return;
            }
            let i = (v & 0xffff_ffff) as usize;
            if i >= self.n.load(Ordering::Relaxed) {
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
            body(i);
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    fn follow(&self) {
        let mut seen = 0u32;
        while !self.exit.load(Ordering::Acquire) {
            let e = self.epoch.load(Ordering::Acquire);
            if e == seen {
                std::hint::spin_loop();
                continue;
            }
            seen = e;
            self.work(e);
        }
    }

    fn run(&self, n: usize, body: Body) {
        let e = self.epoch.load(Ordering::Relaxed) + 1;
        let body_ref: Body = body;
        self.body.store(&body_ref as *const Body as usize, Ordering::Relaxed);
        self.n.store(n, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        self.claim.store(u64::from(e) << 32, Ordering::Release);
        self.epoch.store(e, Ordering::Release);
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
    let team = Team { claim: AtomicU64::new(0), epoch: AtomicU32::new(0), n: AtomicUsize::new(0), done: AtomicUsize::new(0), body: AtomicUsize::new(0), exit: AtomicBool::new(false) };
    let f = Mutex::new(Some(f));
    let result = Mutex::new(None);
    rayon::broadcast(|ctx| {
        if ctx.index() == 0 {
            let f = f.lock().unwrap().take().expect("leader runs once");
            TEAM.with(|t| t.set(&team));
            let r = f();
            TEAM.with(|t| t.set(std::ptr::null()));
            team.exit.store(true, Ordering::Release);
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
