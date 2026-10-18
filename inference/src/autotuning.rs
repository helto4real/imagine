//! Default in-process policy for Kitchen's two approved INT8 Triton autotuners.
//!
//! D-05-INT8 and D-05-INT8-ROW permit independently measured winners, not changed
//! policy/numerics. Each kernel requires its own instance/cache, as upstream.
//! General JIT specialization, disk-cache options and other autotuners are not
//! implemented by this module and must not be silently routed through it.
use crate::native::{CudaEvent, Error, Kitchen};
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Config {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub group_size_m: u32,
    pub stages: u32,
    pub warps: u32,
}
pub const INT8_CONFIGS: [Int8Config; 6] = [
    Int8Config {
        block_m: 128,
        block_n: 256,
        block_k: 64,
        group_size_m: 8,
        stages: 3,
        warps: 8,
    },
    Int8Config {
        block_m: 64,
        block_n: 256,
        block_k: 32,
        group_size_m: 8,
        stages: 4,
        warps: 4,
    },
    Int8Config {
        block_m: 128,
        block_n: 128,
        block_k: 32,
        group_size_m: 8,
        stages: 4,
        warps: 4,
    },
    Int8Config {
        block_m: 128,
        block_n: 64,
        block_k: 32,
        group_size_m: 8,
        stages: 4,
        warps: 4,
    },
    Int8Config {
        block_m: 64,
        block_n: 128,
        block_k: 32,
        group_size_m: 8,
        stages: 4,
        warps: 4,
    },
    Int8Config {
        block_m: 128,
        block_n: 32,
        block_k: 32,
        group_size_m: 8,
        stages: 4,
        warps: 4,
    },
];

/// Source key: (m,n,k), followed by dtype strings of tensor args in call order.
/// Strides, contents, device and bias values are intentionally not added.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Int8Key {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub pointer_dtypes: [String; 6],
}
pub enum BenchFailure<E> {
    OutOfResources,
    CompileTimeAssertionFailure,
    PtxasError,
    Unexpected(E),
}
#[derive(Default)]
/// One instance per original kernel, never shared between tensorwise/per-row.
pub struct Int8Autotuner {
    cache: HashMap<Int8Key, usize>,
    pub best_config: Option<usize>,
    /// The reference retains the last cold selection's timings across cache hits.
    pub last_timings: Option<[[f64; 3]; 6]>,
}
impl Int8Autotuner {
    /// The callback implements the unchanged benchmark procedure. Only the three
    /// reference-caught candidate failures map to infinities. Final execution is
    /// deliberately separate: the reference launches the selected candidate once
    /// after selection on both cold and warm calls, and propagates its failures.
    pub fn select<E>(
        &mut self,
        key: Int8Key,
        mut bench: impl FnMut(usize, &Int8Config) -> Result<[f64; 3], BenchFailure<E>>,
    ) -> Result<usize, E> {
        let index = if let Some(index) = self.cache.get(&key) {
            *index
        } else {
            let mut timings = [[0.; 3]; 6];
            for (index, config) in INT8_CONFIGS.iter().enumerate() {
                timings[index] = match bench(index, config) {
                    Ok(value) => value,
                    Err(
                        BenchFailure::OutOfResources
                        | BenchFailure::CompileTimeAssertionFailure
                        | BenchFailure::PtxasError,
                    ) => [f64::INFINITY; 3],
                    Err(BenchFailure::Unexpected(error)) => return Err(error),
                };
            }
            // Python min compares the timing lists lexicographically and keeps
            // the first candidate on a tie, including when all are infinities.
            let mut best = 0;
            for i in 1..6 {
                if python_less(&timings[i], &timings[best]) {
                    best = i;
                }
            }
            self.cache.insert(key, best);
            self.last_timings = Some(timings);
            best
        };
        self.best_config = Some(index);
        Ok(index)
    }
    pub(crate) fn cached_index(&self, key: &Int8Key) -> Option<usize> {
        self.cache.get(key).copied()
    }
    pub(crate) fn publish_index(&mut self, key: Int8Key, index: usize) {
        self.cache.insert(key, index);
    }
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
fn python_less(a: &[f64; 3], b: &[f64; 3]) -> bool {
    for (x, y) in a.iter().zip(b) {
        if x == y {
            continue;
        }
        return x < y;
    }
    false
}

/// Operations of triton.testing.do_bench, exposed for differential trace tests.
pub trait BenchmarkDevice {
    type Event;
    type Cache;
    fn synchronize(&self) -> Result<(), Error>;
    fn cache(&self) -> Result<Self::Cache, Error>;
    fn clear(&self, cache: &mut Self::Cache) -> Result<(), Error>;
    fn event(&self) -> Result<Self::Event, Error>;
    fn record(&self, event: &mut Self::Event) -> Result<(), Error>;
    fn elapsed(&self, start: &Self::Event, end: &Self::Event) -> Result<f64, Error>;
}
pub struct CudaBenchmark<'a> {
    pub native: &'a Kitchen,
    pub device: usize,
}
impl BenchmarkDevice for CudaBenchmark<'_> {
    type Event = CudaEvent;
    type Cache = Tensor;
    fn synchronize(&self) -> Result<(), Error> {
        self.native.synchronize_device(self.device)
    }
    fn cache(&self) -> Result<Tensor, Error> {
        Ok(Tensor::f_empty(
            [256 * 1024 * 1024 / 4],
            (Kind::Int, Device::Cuda(self.device)),
        )?)
    }
    fn clear(&self, cache: &mut Tensor) -> Result<(), Error> {
        let _ = cache.f_zero_()?;
        Ok(())
    }
    fn event(&self) -> Result<CudaEvent, Error> {
        self.native.timing_event()
    }
    fn record(&self, event: &mut CudaEvent) -> Result<(), Error> {
        event.record_current(self.device)
    }
    fn elapsed(&self, start: &CudaEvent, end: &CudaEvent) -> Result<f64, Error> {
        start.elapsed_ms(end)
    }
}
/// Default warmup=25 ms, rep=100 ms and the autotuner's (0.5,0.2,0.8)
/// quantiles. The INT8 reference has no reset/restore/gradient hooks or pruning.
pub fn do_bench(
    device: &impl BenchmarkDevice,
    mut launch: impl FnMut() -> Result<(), Error>,
) -> Result<[f64; 3], Error> {
    launch()?;
    device.synchronize()?;
    let mut cache = device.cache()?;
    let mut start = device.event()?;
    let mut end = device.event()?;
    device.record(&mut start)?;
    for _ in 0..5 {
        device.clear(&mut cache)?;
        launch()?;
    }
    device.record(&mut end)?;
    device.synchronize()?;
    let estimate = device.elapsed(&start, &end)? / 5.;
    let count = |time: f64| -> Result<usize, Error> {
        let value = (time / estimate).trunc();
        if !value.is_finite() {
            return Err(Error::Native(
                "cannot convert float infinity or NaN to integer".into(),
            ));
        }
        if value < 1. {
            Ok(1)
        } else if value < usize::MAX as f64 {
            Ok(value as usize)
        } else {
            Err(Error::Native(
                "benchmark iteration count exceeds address space".into(),
            ))
        }
    };
    let warmup = count(25.)?;
    let repeat = count(100.)?;
    let mut starts = (0..repeat)
        .map(|_| device.event())
        .collect::<Result<Vec<_>, _>>()?;
    let mut ends = (0..repeat)
        .map(|_| device.event())
        .collect::<Result<Vec<_>, _>>()?;
    for _ in 0..warmup {
        launch()?;
    }
    for i in 0..repeat {
        device.clear(&mut cache)?;
        device.record(&mut starts[i])?;
        launch()?;
        device.record(&mut ends[i])?;
    }
    device.synchronize()?;
    let times = starts
        .iter()
        .zip(&ends)
        .map(|(s, e)| device.elapsed(s, e))
        .collect::<Result<Vec<_>, _>>()?;
    quantiles(&times)
}
pub fn quantiles(times: &[f64]) -> Result<[f64; 3], Error> {
    if times.is_empty() {
        return Err(Error::Native("empty benchmark sample".into()));
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok([0.5, 0.2, 0.8].map(|q| {
        let point = q * (sorted.len() - 1) as f64;
        let lower = point.floor() as usize;
        let upper = point.ceil() as usize;
        let t = point - lower as f64;
        (1. - t) * sorted[lower] + t * sorted[upper]
    }))
}

/// The original wrapper supplies only keyword arguments to each JIT kernel.
/// Its mutable `nargs` starts absent, becomes an empty mapping during a call,
/// remains populated after failure and is set to None only after success.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Int8ArgumentsState {
    #[default]
    Absent,
    EmptyMapping,
    None,
}
#[derive(Default)]
pub(crate) struct Int8RuntimeState {
    pub policy: Int8Autotuner,
    pub arguments: Int8ArgumentsState,
    pub bench_time: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct Int8RuntimeSnapshot {
    pub arguments: Int8ArgumentsState,
    pub cache_entries: usize,
    pub best_config: Option<usize>,
    pub last_timings: Option<[[f64; 3]; 6]>,
    pub bench_time: Option<f64>,
}
impl Int8RuntimeState {
    pub fn snapshot(&self) -> Int8RuntimeSnapshot {
        Int8RuntimeSnapshot {
            arguments: self.arguments,
            cache_entries: self.policy.cache.len(),
            best_config: self.policy.best_config,
            last_timings: self.policy.last_timings,
            bench_time: self.bench_time,
        }
    }
}
