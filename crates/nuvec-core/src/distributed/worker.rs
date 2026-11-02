//! A group's background worker: runs the collectives of GPU streams in program order, so
//! the evaluating thread never waits for the GPU or the network at a collective (MLX's
//! communication stream).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
pub(super) struct Worker {
    state: Mutex<State>,
    /// Woken when a job is queued or the worker shuts down.
    ready: Condvar,
    /// Woken when the queue has drained.
    idle: Condvar,
    /// What the GPU work after a job waits for: raised to each job's value when it is done.
    #[cfg(all(target_os = "macos", not(miri)))]
    pub(super) event: std::sync::OnceLock<nuvec_gpu::Event>,
}

#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    /// A job is running.
    busy: bool,
    started: bool,
    shut_down: bool,
    /// The value of the last job queued.
    issued: u64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Worker {
    /// Locks `order` once every queued job has run, so the caller's collective goes on the
    /// wire after them (no job starts before the caller holds the lock).
    pub(super) fn idle_order<'a>(&self, order: &'a Mutex<()>) -> MutexGuard<'a, ()> {
        let mut state = lock(&self.state);
        while state.busy || !state.jobs.is_empty() {
            state = self
                .idle
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        lock(order)
    }

    /// Queues `job(value)`, which runs on the worker thread after every job queued before
    /// it; returns its value (one more than the previous job's).
    pub(super) fn submit(self: &Arc<Self>, job: impl FnOnce(u64) + Send + 'static) -> u64 {
        let mut state = lock(&self.state);
        state.issued += 1;
        let value = state.issued;
        state.jobs.push_back(Box::new(move || job(value)));
        if !state.started {
            state.started = true;
            let worker = self.clone();
            std::thread::Builder::new()
                .name("nuvec-collectives".into())
                .spawn(move || worker.run())
                .expect("spawn the collectives thread");
        }
        self.ready.notify_one();
        value
    }

    fn run(self: Arc<Self>) {
        loop {
            let job = {
                let mut state = lock(&self.state);
                loop {
                    if let Some(job) = state.jobs.pop_front() {
                        state.busy = true;
                        break job;
                    }
                    if state.shut_down {
                        return;
                    }
                    state = self
                        .ready
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            };
            job();
            let mut state = lock(&self.state);
            state.busy = false;
            if state.jobs.is_empty() {
                self.idle.notify_all();
            }
        }
    }

    /// Ends the thread once the queue is empty (jobs hold their group, so none is left
    /// when the group is dropped).
    pub(super) fn shut_down(&self) {
        lock(&self.state).shut_down = true;
        self.ready.notify_one();
    }

    /// The worker's event, created on `device` at first use.
    #[cfg(all(target_os = "macos", not(miri)))]
    pub(super) fn event(&self, device: &nuvec_gpu::Device) -> crate::Result<&nuvec_gpu::Event> {
        if let Some(event) = self.event.get() {
            return Ok(event);
        }
        let event =
            nuvec_gpu::Event::new(device).map_err(|e| crate::Error::Backend(e.to_string()))?;
        Ok(self.event.get_or_init(|| event))
    }
}
