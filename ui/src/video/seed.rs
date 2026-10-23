//! Advance the widget after confirmed admission, once per idempotent submission.
use super::*;
use imagine_video_contract::generation::{Defaults, SeedControl};
#[derive(Clone)]
pub(super) struct Advance {
    project: ProjectId,
    seed: u64,
    control: SeedControl,
    next: u64,
}
/// Choose once when preparing either a direct submission or an optional review.
pub(super) fn prepare(project: &VideoProject, random: u64) -> (u64, Option<Advance>) {
    let seed = project.generation.as_ref().map_or(42, |g| {
        if g.control_after_generate.is_none() && g.random_seed {
            random & MAX_INTEGER
        } else {
            g.seed
        }
    });
    (seed, Advance::capture(project, random))
}
impl Advance {
    pub fn capture(project: &VideoProject, random_bits: u64) -> Option<Self> {
        let generation = project.generation.as_ref()?;
        let control = generation.control_after_generate?;
        let next = control.next(generation.seed, random_bits);
        (next != generation.seed).then(|| Self {
            project: project.id.clone(),
            seed: generation.seed,
            control,
            next,
        })
    }
    pub fn apply(self, project: &VideoProject) -> Option<Defaults> {
        let mut generation = project.generation.clone()?;
        if self.project != project.id
            || generation.seed != self.seed
            || generation.control_after_generate != Some(self.control)
        {
            return None;
        }
        generation.seed = self.next;
        Some(generation)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn project() -> VideoProject {
        VideoProject::create(CreateProject {
            schema_version: 1,
            generation: Some(Defaults {
                seed: 42,
                control_after_generate: Some(SeedControl::Increment),
                ..Default::default()
            }),
            default_model: imagine_video_contract::generation::h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap()
    }
    #[test]
    fn confirmed_admission_updates_only_the_matching_seed_and_project() {
        let mut p = project();
        let advance = Advance::capture(&p, 0).unwrap();
        let next = advance.apply(&p).unwrap();
        assert_eq!(next.seed, 43);
        let stale = Advance::capture(&p, 0).unwrap();
        p.generation = Some(next);
        assert!(stale.apply(&p).is_none());
        let stale = Advance::capture(&p, 0).unwrap();
        p.id = ProjectId("another-own-project".into());
        assert!(stale.apply(&p).is_none());
        let stale = Advance::capture(&p, 0).unwrap();
        p.generation.as_mut().unwrap().control_after_generate = Some(SeedControl::Fixed);
        assert!(stale.apply(&p).is_none());
        assert!(Advance::capture(&p, 0).is_none());
        p.generation = None;
        assert!(Advance::capture(&p, 0).is_none());
    }
}
