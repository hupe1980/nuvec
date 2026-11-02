//! Persistent helper threads for the ring's fork-join work (a sender beside a receiver, the
//! parts of a split reduction), as MLX's ring keeps its thread pool: `thread::scope` without
//! a new thread per call. A thread is added only when every existing one is busy, so nested
//! joins never wait for a free thread.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct Pool {
    state: Mutex<State>,
    ready: Condvar,
}

#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    /// Threads waiting for a job.
    idle: usize,
}

fn pool() -> &'static Arc<Pool> {
    static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
    POOL.get_or_init(Arc::default)
}

impl Pool {
    fn submit(self: &Arc<Self>, job: Job) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.jobs.push_back(job);
        if state.jobs.len() > state.idle {
            let pool = self.clone();
            std::thread::Builder::new()
                .name("nuvec-ring".into())
                .spawn(move || pool.work())
                .expect("spawn a ring thread");
        }
        self.ready.notify_one();
    }

    fn work(&self) {
        loop {
            let job = {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                loop {
                    if let Some(job) = state.jobs.pop_front() {
                        break job;
                    }
                    state.idle += 1;
                    state = self
                        .ready
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                    state.idle -= 1;
                }
            };
            job();
        }
    }
}

/// Runs `side` on a helper thread while `main` runs on this one; returns both results once
/// both have finished (a panic in either is resumed here, after both finished).
#[allow(unsafe_code)]
pub(crate) fn join<'a, A: Send + 'a, B>(
    side: impl FnOnce() -> A + Send + 'a,
    main: impl FnOnce() -> B,
) -> (A, B) {
    type Slot<A> = (Mutex<Option<std::thread::Result<A>>>, Condvar);
    let slot: Arc<Slot<A>> = Arc::new((Mutex::new(None), Condvar::new()));
    let job: Box<dyn FnOnce() + Send + 'a> = Box::new({
        let slot = slot.clone();
        move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(side));
            *slot.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
            slot.1.notify_one();
        }
    });
    // SAFETY: the job borrows data of lifetime `'a`; this function returns (or resumes a
    // panic) only after the job has run to completion (it stores its result last), so no
    // borrow outlives `'a`. A panic in `main` is caught and waits for the job too.
    let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Job>(job) };
    pool().submit(job);
    let main = std::panic::catch_unwind(std::panic::AssertUnwindSafe(main));
    let mut done = slot.0.lock().unwrap_or_else(PoisonError::into_inner);
    let side = loop {
        if let Some(result) = done.take() {
            break result;
        }
        done = slot.1.wait(done).unwrap_or_else(PoisonError::into_inner);
    };
    drop(done);
    match (side, main) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(p), _) | (_, Err(p)) => std::panic::resume_unwind(p),
    }
}
