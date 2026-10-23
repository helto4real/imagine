//! Bounded executable H3 sampling intent; shared by the product and native worker.
use serde::{Deserialize, Serialize};

pub const MAX_STEPS: u32 = 10_000;
pub const SCHEDULERS: [&str; 9] = [
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
];
pub fn full_denoise() -> f64 {
    1.
}
pub fn base_steps() -> u32 {
    4
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub steps: u32,
    pub sampler: String,
    pub scheduler: String,
    #[serde(default = "full_denoise")]
    pub denoise: f64,
    #[serde(default)]
    pub sigmas: Vec<f32>,
}
impl Default for Stage {
    fn default() -> Self {
        Self {
            steps: 20,
            sampler: "res_multistep".into(),
            scheduler: "simple".into(),
            denoise: 1.,
            sigmas: vec![],
        }
    }
}
impl Stage {
    pub fn intervals(&self) -> u32 {
        if self.sigmas.is_empty() {
            self.steps
        } else {
            self.sigmas.len().saturating_sub(1) as u32
        }
    }
    pub fn valid(&self) -> bool {
        self.steps > 0
            && self.steps <= MAX_STEPS
            && matches!(self.sampler.as_str(), "res_multistep" | "euler")
            && SCHEDULERS.contains(&self.scheduler.as_str())
            && self.denoise.is_finite()
            && self.denoise > 0.
            && self.denoise <= 1.
            && (f64::from(self.steps) / self.denoise) <= f64::from(MAX_STEPS)
            && (self.sigmas.is_empty() || valid_sigmas(&self.sigmas))
    }
}
pub fn valid_sigmas(sigmas: &[f32]) -> bool {
    sigmas.len() >= 2
        && sigmas.len() <= MAX_STEPS as usize + 1
        && sigmas
            .iter()
            .all(|value| value.is_finite() && *value >= 0. && *value <= 1.)
        && sigmas.windows(2).all(|pair| pair[1] <= pair[0])
        && sigmas.windows(2).any(|pair| pair[1] < pair[0])
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handoff {
    ContinueLatent,
    ReNoiseVideoPrediction,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondStage {
    pub sampling: Stage,
    pub profile: super::Profile,
    pub handoff: Handoff,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub first: Stage,
    #[serde(default = "base_steps")]
    pub base_steps_to_run: u32,
    pub second: Option<SecondStage>,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            first: Stage::default(),
            base_steps_to_run: base_steps(),
            second: None,
        }
    }
}
impl Sampling {
    pub fn total_steps(&self) -> u32 {
        match &self.second {
            Some(second) => {
                (if self.first.sigmas.is_empty() {
                    self.base_steps_to_run
                } else {
                    self.first.intervals()
                }) + second.sampling.intervals()
            }
            None => self.first.intervals(),
        }
    }
    pub fn valid(&self) -> bool {
        if !self.first.valid() {
            return false;
        }
        match &self.second {
            None => self.first.sigmas.last().is_none_or(|sigma| *sigma == 0.),
            Some(second) => {
                second.profile != super::Profile::ShortAvNoReferenceV1
                    && second.sampling.valid()
                    && (if self.first.sigmas.is_empty() {
                        self.base_steps_to_run > 0 && self.base_steps_to_run < self.first.steps
                    } else {
                        self.first
                            .sigmas
                            .last()
                            .is_some_and(|sigma| *sigma > 0. && *sigma < 1.)
                    })
                    && second
                        .sampling
                        .sigmas
                        .last()
                        .is_none_or(|sigma| sigma.abs() <= 1e-6)
                    && (self.first.sigmas.is_empty()
                        || second.sampling.sigmas.is_empty()
                        || (self.first.sigmas.last().unwrap() - second.sampling.sigmas[0]).abs()
                            <= 1e-6)
            }
        }
    }
}
