//! Synthetic schedule math; no Runtime, checkpoint, native library or CUDA initialization.
use imagine_inference::{
    models::minimax_h3::{dit, sampling as h3},
    sampling::{
        model_sampling::{Flavor, ModelSampling},
        schedules::ScheduleModel,
    },
};
use serde_json::json;
use tch::{Device, Kind, Tensor};
fn values(t: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(t.to_kind(Kind::Double).view([-1])).unwrap()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(1);
    let mut cases = vec![];
    for (video, audio) in [(12., 3.), (9.5, 2.25), (1., 1.), (0.5, 2.)] {
        let mut schedule = ModelSampling {
            flavor: Flavor::AudioVideo,
            config: h3::schedule_config(video, audio),
            sigmas: Tensor::new(),
            log_sigmas: None,
            num_timesteps: 0,
            cosine_s: None,
            initial_alpha: None,
            skip_steps: 1,
        };
        let time = Tensor::arange_start(1, 1001, (Kind::Int64, Device::Cpu)) / 1000. * 1000.;
        schedule.sigmas = schedule.sigma(&time)?;
        let sigmas = h3::simple_sigmas(&schedule, 20)?;
        let audio_sigmas = dit::time_shift(&sigmas, video, audio)?;
        cases.push(json!({"video_shift":video,"audio_shift":audio,"video_sigmas":values(&sigmas),"audio_sigmas":values(&audio_sigmas),"audio_scale":schedule.audio_scale()?}));
    }
    println!("{}", json!({"device":"cpu","cases":cases}));
    Ok(())
}
