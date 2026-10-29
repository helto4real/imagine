//! Explicit ignored GPU component test. No text, jobs, decoded media or dumps.
use super::*;
use std::{
    cell::{Cell, RefCell},
    fs::OpenOptions,
    io::Write,
};

#[test]
#[ignore = "Explicit own synthetic H3 DiT test; requires coordinator lease held by its harness"]
fn named_sampling_with_manual_sigmas_on_real_dit() {
    let result_path = PathBuf::from(std::env::var_os("IMAGINE_CP40_SAMPLER_RESULT").unwrap());
    let stage = Cell::new("runtime_initialization");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        numeric_probe(&result_path, &stage)
    }));
    if let Err(panic) = result {
        if !result_path.exists() {
            let value = serde_json::json!({"result":{"passed":false,"panic":true,"stage":stage.get()},
                "scope":"own numeric component test phase only; panic/console content is not read"});
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(result_path)
                .unwrap();
            writeln!(output, "{value}").unwrap();
        }
        // Keep the original panic and its untouched console diagnostics.
        std::panic::resume_unwind(panic);
    }
}
fn numeric_probe(result_path: &std::path::Path, stage: &Cell<&'static str>) {
    let own =
        std::path::Path::new("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01");
    assert!(result_path.starts_with(own) && !result_path.exists());
    assert_eq!(
        std::env::var("IMAGINE_CP40_EXCLUSIVE_LEASE").unwrap(),
        "held"
    );
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let rt = Runtime::frozen_partial(
        unsafe {
            Kitchen::load(std::path::Path::new(
                &std::env::var("IMAGINE_KITCHEN_LIBRARY").unwrap(),
            ))
        }
        .unwrap(),
    )
    .unwrap();
    bind_comfy_layouts(&rt);
    rt.backends.policy.disable("triton");
    stage.set("aimdo_initialization");
    let aimdo = unsafe {
        Aimdo::load(std::path::Path::new(
            &std::env::var("IMAGINE_AIMDO_LIBRARY").unwrap(),
        ))
    }
    .unwrap();
    let model_root = PathBuf::from(std::env::var_os("IMAGINE_CP40_MODEL_ROOT").unwrap());
    let models = Models::new(model_root, false, Default::default(), Default::default()).unwrap();
    let shape = p::Shape {
        width: 416,
        height: 224,
        frames: 39,
        fps: 24,
    };
    let reports = RefCell::new(Vec::<p::progress::Report>::new());
    let mut notify = |notice| {
        if let Notice::Step(report) = notice {
            reports.borrow_mut().push(report);
        }
        Ok(())
    };
    let started = Instant::now();
    stage.set("component_load");
    let result = component(
        &rt,
        &aimdo,
        &models,
        Component::Dit,
        shape,
        &mut notify,
        |params, schedule, exec| {
            stage.set("numeric_input_construction");
            schedule.sigmas = params["model_sampling.sigmas"].as_tensor()?;
            let config = h3::dit::Config::selected()?;
            let text_dim = config.text_dim;
            let model = h3::dit::Denoiser::from_parameters(config, params)?;
            let base = Tensor::f_zeros(
                [1, 32, shape.video_latent_t(), 14, 26],
                (Kind::Float, Device::Cuda(0)),
            )?;
            let latent =
                h3::streams::NestedTensor::new(h3::preprocessing::empty_streams(&base)?.into());
            let noise = latent.prepare_noise(42, None)?;
            // Numeric conditioning fixture, not tokenized text or saved conditioning.
            let context = Tensor::f_arange(4 * text_dim, (Kind::Float, Device::Cuda(0)))?
                .f_mul_scalar(0.001)?
                .f_sin()?
                .f_mul_scalar(0.05)?
                .f_reshape([1, 4, text_dim])?;
            let tags = h3::preprocessing::token_tags(4, &[])?.f_to_device(Device::Cuda(0))?;
            let mut rows = Vec::new();
            for (sampler, curve) in [
                ("res_multistep", vec![1_f32, 0.6, 0.]),
                ("heun", vec![1.2_f32, 0.6, 0.]),
                // CFG++ uses log((1-sigma)/sigma) for CONST. Exercise its
                // finite domain; the separate Heun case tests sigma above one.
                ("euler_cfg_pp", vec![0.9_f32, 0.6, 0.]),
                ("dpm_fast", vec![1_f32, 0.8, 0.6, 0.3, 0.]),
            ] {
                stage.set(sampler);
                let spec = p::sampling::Stage {
                    sampler: sampler.into(),
                    sigmas: curve.clone(),
                    steps: 0,
                    denoise: -1.,
                    scheduler: "unused manual metadata".into(),
                };
                assert!(spec.valid());
                let sigmas = Tensor::f_from_slice(&curve)?.f_to_device(Device::Cuda(0))?;
                reports.borrow_mut().clear();
                exec.sampler_progress = p::progress::Tracker::new(spec.intervals());
                exec.sampler_progress.begin_stage(0);
                let start = Instant::now();
                let output = unsafe {
                    h3::sampling::sample_named_cfg1(
                        sampler,
                        &rt,
                        &model,
                        schedule,
                        h3::sampling::EulerRequest {
                            latent: &latent,
                            noise: &noise,
                            sigmas: &sigmas,
                            context: &context,
                            payload: h3::dit::Payload {
                                text_tags: Some(&tags),
                                seed: 42,
                                ..Default::default()
                            },
                            denoise_mask: None,
                            kind: Kind::BFloat16,
                        },
                        exec,
                    )
                }?;
                if let Some(report) = exec.sampler_progress.finish_stage() {
                    (exec.notify)(Notice::Step(report))?;
                }
                rt.native.synchronize_device(0)?;
                let values = output.unbind();
                let mut streams = Vec::new();
                for value in values.borrow().iter() {
                    let tensor = value
                        .as_tensor()?
                        .f_to_device(Device::Cpu)?
                        .f_to_kind(Kind::Double)?;
                    let finite = tensor.f_isfinite()?.f_all()?.int64_value(&[]) == 1;
                    streams.push(serde_json::json!({"shape":tensor.size(),
                    "finite":finite,
                    "rms":tensor.f_square()?.f_mean(Kind::Double)?.f_sqrt()?.double_value(&[]),
                    "peak":tensor.f_abs()?.f_max()?.double_value(&[])}));
                }
                let progress = reports.borrow();
                let progress_ok = progress
                    .last()
                    .is_some_and(|r| r.completed == 2 && r.total == 2);
                let passed = progress_ok && streams.iter().all(|s| s["finite"] == true);
                rows.push(serde_json::json!({"sampler":sampler,"sigmas":curve,"passed":passed,
                "planned_intervals":spec.intervals(),"seconds":start.elapsed().as_secs_f64(),
                "progress":progress.iter().map(|r|[r.completed,r.total]).collect::<Vec<_>>(),"streams":streams}));
                if !passed {
                    break;
                }
            }
            stage.set("component_unload");
            Ok(rows)
        },
    );
    let memory_after = rt.native.cuda_memory(0).ok();
    let report = match &result {
        Ok(rows) => {
            serde_json::json!({"passed":rows.len()==4 && rows.iter().all(|r|r["passed"]==true),"variants":rows})
        }
        Err(code) => serde_json::json!({"passed":false,"component_error":code,"stage":stage.get()}),
    };
    let report = serde_json::json!({"result":report,"seconds":started.elapsed().as_secs_f64(),
        "device":"cuda:0","kind":"bfloat16","shape":[416,224,39,24],
        "conditioning":"four deterministic numeric tokens, no text encoding",
        "memory_after":memory_after,"no_text_jobs_decoded_media_or_latent_dumps":true,
        "scope":"real DiT named sampler/manual SIGMAS/callback component smoke; no VAE/export or full Python pipeline parity"});
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(result_path)
        .unwrap();
    writeln!(output, "{report}").unwrap();
    assert!(
        report["result"]["passed"] == true,
        "Own synthetic DiT component failed; structured code saved, console unread"
    );
}

#[test]
fn cfgpp_manual_sigma_domain_retains_unclamped_reference_algebra() {
    let sigma = Tensor::from_slice(&[1.2_f32, 1., 0.9]);
    let model = crate::sampling::parameterization::Parameterization {
        prediction: crate::sampling::parameterization::Prediction::Const,
        sigma_data: 1.,
        noise_scale: 1.,
    };
    let log_snr = crate::sampling::solvers::half_log_snr(&sigma, model).unwrap();
    assert!(log_snr.double_value(&[0]).is_nan());
    assert_eq!(log_snr.double_value(&[1]), f64::NEG_INFINITY);
    assert!(log_snr.double_value(&[2]).is_finite());
}
