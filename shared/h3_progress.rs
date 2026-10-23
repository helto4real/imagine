//! Sampling callbacks are observations, not a fixed solver iteration contract.
//! Adaptive solvers may use more or fewer callbacks than sigma intervals.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    pub completed: u32,
    pub total: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Tracker {
    report: Report,
    offset: u32,
    remaining: u32,
}
impl Tracker {
    pub fn new(planned: u32) -> Self {
        Self {
            report: Report {
                completed: 0,
                total: planned,
            },
            offset: 0,
            remaining: 0,
        }
    }
    /// The following stages retain their planned count until they execute.
    pub fn begin_stage(&mut self, following_intervals: u32) {
        self.offset = self.report.completed;
        self.remaining = following_intervals;
    }
    pub fn step(&self, index: u32) -> Option<u32> {
        self.offset.checked_add(index)?.checked_add(1)
    }
    pub fn observe(&mut self, index: u32) -> Option<Report> {
        let completed = self.step(index)?;
        if completed <= self.report.completed {
            return None;
        }
        let minimum = completed.checked_add(self.remaining)?;
        self.report = Report {
            completed,
            total: self.report.total.max(minimum),
        };
        Some(self.report)
    }
    /// A completed invocation makes its actual callback extent known. A total
    /// correction is allowed without inventing an additional model callback.
    pub fn finish_stage(&mut self) -> Option<Report> {
        let total = self.report.completed.checked_add(self.remaining)?;
        if self.report.completed == 0 || total == self.report.total {
            return None;
        }
        self.report.total = total;
        Some(self.report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_stages_preserve_existing_counts() {
        let mut tracker = Tracker::new(4);
        tracker.begin_stage(2);
        assert_eq!(
            tracker.observe(0),
            Some(Report {
                completed: 1,
                total: 4
            })
        );
        assert_eq!(
            tracker.observe(1),
            Some(Report {
                completed: 2,
                total: 4
            })
        );
        assert_eq!(tracker.finish_stage(), None);
        tracker.begin_stage(0);
        assert_eq!(
            tracker.observe(0),
            Some(Report {
                completed: 3,
                total: 4
            })
        );
        assert_eq!(
            tracker.observe(1),
            Some(Report {
                completed: 4,
                total: 4
            })
        );
        assert_eq!(tracker.finish_stage(), None);
    }
    #[test]
    fn adaptive_extent_and_fast_stage_define_the_next_offset() {
        let mut tracker = Tracker::new(8);
        tracker.begin_stage(4);
        for i in 0..6 {
            tracker.observe(i).unwrap();
        }
        assert_eq!(tracker.finish_stage(), None);
        tracker.begin_stage(0);
        assert_eq!(
            tracker.observe(0),
            Some(Report {
                completed: 7,
                total: 10
            })
        );
        assert_eq!(
            tracker.observe(1),
            Some(Report {
                completed: 8,
                total: 10
            })
        );
        assert_eq!(
            tracker.finish_stage(),
            Some(Report {
                completed: 8,
                total: 8
            })
        );
    }
    #[test]
    fn repeated_or_skipped_callback_labels_do_not_abort_sampling() {
        let mut tracker = Tracker::new(4);
        tracker.begin_stage(0);
        assert_eq!(
            tracker.observe(0),
            Some(Report {
                completed: 1,
                total: 4
            })
        );
        assert_eq!(tracker.observe(0), None);
        assert_eq!(
            tracker.observe(3),
            Some(Report {
                completed: 4,
                total: 4
            })
        );
        assert_eq!(tracker.observe(u32::MAX), None);
        assert_eq!(tracker.finish_stage(), None);
    }
}
