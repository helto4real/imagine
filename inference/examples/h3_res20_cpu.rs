//! CPU-only deterministic parity probe. No Runtime, checkpoint or CUDA initialization.
use imagine_inference::{
    models::{
        minimax_h3::sampling as h3,
        qwen_image21::{EulerSettings, EulerStep},
    },
    native::Error,
    sampling::{
        model_sampling::{Config, Flavor, ModelSampling},
        solvers::{self, Execution, NoiseKind, Prediction},
    },
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
fn values(x: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(x.to_kind(Kind::Double).view([-1])).unwrap()
}
#[derive(Default)]
struct Mock {
    trace: Vec<Value>,
}
impl Execution for Mock {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg_pp: bool) -> Result<Prediction, Error> {
        assert!(!cfg_pp);
        assert_eq!(x.device(), Device::Cpu);
        // Nonlinear, cross-stream predictor exercises multistep history; no model or RNG.
        let d = x
            .f_mul_scalar(0.37)?
            .f_add(&x.f_flip([-1])?.f_mul_scalar(0.13)?)?
            .f_add(&sigma.f_reshape([-1, 1])?.f_mul_scalar(0.19)?)?
            .f_tanh()?;
        Ok(Prediction {
            denoised: d,
            uncond_denoised: None,
            cond_denoised: None,
        })
    }
    fn noise(&mut self, _: &Tensor, _: &Tensor, _: &Tensor, _: NoiseKind) -> Result<Tensor, Error> {
        panic!("deterministic res_multistep must not request stochastic noise")
    }
    fn callback(&mut self, s: EulerStep<'_>) -> Result<(), Error> {
        self.trace.push(
            json!({"i":s.i,"sigma":values(s.sigma),"x":values(s.x),"denoised":values(s.denoised)}),
        );
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(1);
    let mut schedule = ModelSampling {
        flavor: Flavor::AudioVideo,
        config: Config {
            shift: 12.,
            audio_shift: Some(3.),
            ..Default::default()
        },
        sigmas: Tensor::new(),
        log_sigmas: None,
        num_timesteps: 0,
        cosine_s: None,
        initial_alpha: None,
        skip_steps: 1,
    };
    use imagine_inference::sampling::schedules::ScheduleModel;
    // Same table initialization as ModelSampling::new, using only CPU tensors.
    let t = Tensor::arange_start(1, 1001, (Kind::Int64, Device::Cpu)) / 1000. * 1000.;
    schedule.sigmas = schedule.sigma(&t)?;
    let sigmas = h3::simple_sigmas(&schedule, 20)?;
    let audio_sigmas = imagine_inference::models::minimax_h3::dit::time_shift(&sigmas, 12., 3.)?;
    use imagine_inference::{
        execution_value::Value as V,
        models::minimax_h3::{flow, streams::NestedTensor},
    };
    let video = Tensor::from_slice(&[-0.5f32, 0.25]);
    let audio = Tensor::from_slice(&[0.1f32, -0.4, 0.75]);
    let streams = NestedTensor::new(vec![V::tensor(video.copy()), V::tensor(audio.copy())]);
    let scaled = flow::process_streams_in(&streams, schedule.audio_scale()?)?;
    let scaled_audio = values(&scaled.unbind().borrow()[1].as_tensor()?);
    let restored = flow::process_streams_out(&scaled, schedule.audio_scale()?)?;
    assert_eq!(
        values(&restored.unbind().borrow()[0].as_tensor()?),
        values(&video)
    );
    assert_eq!(
        values(&restored.unbind().borrow()[1].as_tensor()?),
        values(&audio)
    );
    let x = Tensor::from_slice(&[-0.4f32, 0.2, 0.7, -0.1, 0.4, -0.8, 0.3, 0.5]).reshape([1, 8]);
    let mut cases = Vec::new();
    // Full20, first-step/end-only and truncated schedule test initialization and zero termination.
    for n in [20, 1, 2] {
        let s = if n == 20 {
            sigmas.shallow_clone()
        } else {
            h3::simple_sigmas(&schedule, n)?
        };
        let mut exec = Mock::default();
        let y = solvers::res_multistep(
            &mut exec,
            x.copy(),
            &s,
            h3::parameterization(&schedule),
            0.,
            1.,
            false,
        )?;
        assert_eq!(exec.trace.len(), n as usize);
        let mut euler = Mock::default();
        let ey = solvers::churn(
            &mut euler,
            x.copy(),
            &s,
            EulerSettings::default(),
            solvers::ChurnSolver::Euler,
        )?;
        cases.push(json!({"steps":n,"sigmas":values(&s),"trace":exec.trace,"out":values(&y),"euler":values(&ey)}));
    }
    println!(
        "{}",
        json!({"device":"cpu","audio_scale":schedule.audio_scale()?,"audio_sigmas":values(&audio_sigmas),"scaled_audio":scaled_audio,"cases":cases})
    );
    Ok(())
}
