use imagine_inference::{autotuning::*, native::Error};
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};

fn fixtures() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("IMAGINE_AUTOTUNING_POLICY_FIXTURE")
                .expect("Explicit source-driven synthetic policy fixture required"),
        )
        .unwrap(),
    )
    .unwrap()
}
struct ClockDevice {
    trace: RefCell<Vec<Value>>,
    next_event: Cell<usize>,
}
impl ClockDevice {
    fn log(&self, value: Value) {
        self.trace.borrow_mut().push(value);
    }
}
impl BenchmarkDevice for ClockDevice {
    type Event = usize;
    type Cache = ();
    fn synchronize(&self) -> Result<(), Error> {
        self.log(json!(["sync"]));
        Ok(())
    }
    fn cache(&self) -> Result<(), Error> {
        self.log(json!(["cache", 256 * 1024 * 1024]));
        Ok(())
    }
    fn clear(&self, _: &mut ()) -> Result<(), Error> {
        self.log(json!(["clear"]));
        Ok(())
    }
    fn event(&self) -> Result<usize, Error> {
        let id = self.next_event.get();
        self.next_event.set(id + 1);
        self.log(json!(["event", id]));
        Ok(id)
    }
    fn record(&self, event: &mut usize) -> Result<(), Error> {
        self.log(json!(["record", event]));
        Ok(())
    }
    fn elapsed(&self, start: &usize, end: &usize) -> Result<f64, Error> {
        self.log(json!(["elapsed", start, end]));
        Ok(if *start == 0 {
            50.
        } else {
            [1., 3., 2., 5., 4., 9., 6., 7., 8., 10.][start - 2]
        })
    }
}
#[test]
fn benchmark_order_counts_and_quantiles_match_original_triton() {
    let reference = fixtures();
    let clock = ClockDevice {
        trace: RefCell::new(vec![]),
        next_event: Cell::new(0),
    };
    let summary = do_bench(&clock, || {
        clock.log(json!(["launch"]));
        Ok(())
    })
    .unwrap();
    assert_eq!(json!(summary), reference["benchmark"]["quantiles"]);
    assert_eq!(
        json!(*clock.trace.borrow()),
        reference["benchmark"]["trace"]
    );
}
fn key(dtype: &str) -> Int8Key {
    Int8Key {
        m: 16,
        n: 128,
        k: 128,
        pointer_dtypes: [
            "torch.int8",
            "torch.int8",
            dtype,
            "torch.float32",
            "torch.float32",
            dtype,
        ]
        .map(str::to_owned),
    }
}
#[test]
fn candidate_order_lexicographic_ties_and_cache_match_original_triton() {
    let reference = fixtures();
    let mut tuner = Int8Autotuner::default();
    let mut trace = vec![];
    let mut winners = vec![];
    for dtype in ["torch.float32", "torch.float32", "torch.float16"] {
        let winner = tuner
            .select(key(dtype), |i, _| {
                trace.push(json!(["launch", i]));
                let timing = &reference["selection"]["timings"][i];
                if timing.is_null() {
                    return Err(BenchFailure::<()>::OutOfResources);
                }
                trace.push(json!(["benchmark", i]));
                Ok(std::array::from_fn(|j| timing[j].as_f64().unwrap()))
            })
            .unwrap();
        trace.push(json!(["launch", winner]));
        winners.push(winner);
    }
    assert_eq!(json!(winners), reference["selection"]["winners"]);
    assert_eq!(json!(trace), reference["selection"]["trace"]);
    assert_eq!(tuner.best_config, Some(3));
}
#[test]
fn only_reference_candidate_failures_are_swallowed() {
    let mut tuner = Int8Autotuner::default();
    assert_eq!(
        tuner.select(key("torch.float32"), |i, _| Err::<[f64; 3], _>(match i {
            0 => BenchFailure::<&str>::OutOfResources,
            1 => BenchFailure::CompileTimeAssertionFailure,
            2 => BenchFailure::PtxasError,
            _ => BenchFailure::Unexpected("native launch failed"),
        })),
        Err("native launch failed")
    );
    assert!(tuner.best_config.is_none());
    assert!(tuner.last_timings.is_none());
    let winner = tuner
        .select(key("torch.float32"), |_, _| {
            Err::<[f64; 3], _>(BenchFailure::<()>::OutOfResources)
        })
        .unwrap();
    assert_eq!(winner, 0); // Original min keeps the first infinity; final launch must still fail.
    assert!(quantiles(&[]).is_err());
}
