use imagine_h3_product_cpu::protocol;
#[path = "../../../src/h3_product/memory_budget.rs"]
mod memory_budget;
#[path = "../../../src/memory_policy.rs"]
mod memory_policy;

use memory_policy::{MemoryPolicy, VramState};
const GIB: f64 = 1073741824.;
fn shape(width: u32, height: u32, frames: u32) -> protocol::Shape {
    protocol::Shape {
        width,
        height,
        frames,
        fps: 24,
    }
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 0.000001,
        "{actual} != {expected}"
    );
}

#[test]
fn packed_av_counts_both_streams_before_batch_and_bf16_scaling() {
    for (s, video_elements, audio_elements) in [
        (shape(384, 256, 39), 110592., 4160.),
        (shape(768, 512, 124), 1363968., 13248.),
        (shape(512, 768, 124), 1363968., 13248.),
    ] {
        let (requested, minimum) = memory_budget::dit(s);
        close(
            minimum,
            (video_elements + audio_elements) * 2. * 0.01 * 0.114 * 1048576.,
        );
        close(requested, minimum * 2.);
    }
}

#[test]
fn short_profile_retains_original_floor_and_request() {
    let p = MemoryPolicy::default();
    let (request, minimum) = memory_budget::dit(shape(384, 256, 39));
    let old = p.requirements(request, None);
    let candidate = p.requirements(request, Some(minimum));
    assert_eq!(old, candidate);
    close(candidate.1 / GIB, 1.190625);
}

#[test]
fn larger_profile_changes_only_minimum_not_requested_reserve() {
    let p = MemoryPolicy::default();
    let (request, minimum) = memory_budget::dit(shape(768, 512, 124));
    let old = p.requirements(request, None);
    let candidate = p.requirements(request, Some(minimum));
    assert_eq!(old.0, candidate.0);
    close(old.1 / GIB, 6.523540);
    close(candidate.1 / GIB, 3.4570825);
    close((old.1 - candidate.1) / GIB, 3.0664575);
    for free in [0., GIB, 4. * GIB, 12. * GIB, 16. * GIB] {
        for loaded in [0., GIB] {
            let before = p.lowvram_budget(free, loaded, old.1);
            let after = p.lowvram_budget(free, loaded, candidate.1);
            assert!(after >= before);
            if free + loaded <= candidate.1 {
                assert_eq!(after, before);
            }
        }
    }
}

#[test]
fn reserve_and_policy_gates_are_preserved() {
    let (request, minimum) = memory_budget::dit(shape(768, 512, 124));
    for reserved in [0., 400. * 1048576., 2. * GIB] {
        let p = MemoryPolicy {
            reserved,
            ..Default::default()
        };
        let (extra, min) = p.requirements(request, Some(minimum));
        close(extra, (0.8 * GIB + reserved).max(request + reserved));
        close(min, (0.8 * GIB + reserved).max(minimum + reserved));
    }
    for state in [
        VramState::Disabled,
        VramState::NoVram,
        VramState::LowVram,
        VramState::NormalVram,
        VramState::HighVram,
        VramState::Shared,
    ] {
        let p = MemoryPolicy {
            vram_state: state,
            ..Default::default()
        };
        assert!(!p.uses_lowvram_budget(true, false));
        assert!(!p.uses_lowvram_budget(false, true));
        assert_eq!(
            p.uses_lowvram_budget(false, false),
            matches!(state, VramState::LowVram | VramState::NormalVram)
        );
        assert!(
            !MemoryPolicy {
                lowvram_available: false,
                ..p
            }
            .uses_lowvram_budget(false, false)
        );
        assert!(!p.disable_smart_memory);
    }
}
