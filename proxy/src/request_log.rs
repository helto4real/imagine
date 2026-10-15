use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};
#[derive(Clone)]
pub struct RequestLog(Arc<Inner>);
struct Inner {
    started: Instant,
    target: String,
    method: String,
    path: String,
    level: i32,
    counter: Arc<AtomicUsize>,
    done: AtomicBool,
}
#[derive(Clone)]
pub struct Streamed;
impl RequestLog {
    pub fn new(
        target: &str,
        method: &str,
        path: &str,
        level: i32,
        counter: Arc<AtomicUsize>,
    ) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(Arc::new(Inner {
            started: Instant::now(),
            target: target.into(),
            method: method.into(),
            path: path.into(),
            level,
            counter,
            done: AtomicBool::new(false),
        }))
    }
    pub fn complete(&self, status: u16) {
        self.finish(
            self.0.level,
            format!(
                "request completed: target={} method={} path={} status={} duration={:.3}s",
                self.0.target,
                self.0.method,
                self.0.path,
                status,
                self.0.started.elapsed().as_secs_f64()
            ),
        )
    }
    pub fn rejected(&self, status: u16, reason: &str) {
        self.finish(
            if status == 503 { 40 } else { 30 },
            format!(
                "request rejected: target={} method={} path={} status={} reason={} duration={:.3}s",
                self.0.target,
                self.0.method,
                self.0.path,
                status,
                reason,
                self.0.started.elapsed().as_secs_f64()
            ),
        )
    }
    pub fn failed(&self, error: &str) {
        self.finish(
            40,
            format!(
                "request failed: target={} method={} path={} status=502 error={} duration={:.3}s",
                self.0.target,
                self.0.method,
                self.0.path,
                error,
                self.0.started.elapsed().as_secs_f64()
            ),
        )
    }
    fn finish(&self, level: i32, message: String) {
        if !self.0.done.swap(true, Ordering::AcqRel) {
            crate::logging::log(level, "st_proxy.service", message);
            self.0.counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}
impl Drop for Inner {
    fn drop(&mut self) {
        if !self.done.load(Ordering::Relaxed) {
            self.counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}
