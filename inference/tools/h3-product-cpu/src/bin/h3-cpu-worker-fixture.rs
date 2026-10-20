//! CPU-only protocol/lifecycle fixture. Never loads a model or GPU library.
use imagine_h3_product_cpu::{
    deployment,
    protocol::{self as p, ErrorCode, Event, Phase, Request},
    wire::{self, Emit, Generator},
};
use std::{collections::BTreeMap, io, path::PathBuf};
struct Fake;
impl Generator for Fake {
    fn generate(&mut self, r: &Request, emit: &mut Emit<'_>) -> Result<(), ErrorCode> {
        if r.prompt == "SYNTHETIC_PRIVATE_PANIC_CANARY" {
            panic!("{}", r.prompt);
        }
        if r.prompt == "SYNTHETIC_WAIT" {
            std::thread::sleep(std::time::Duration::from_secs(30));
        }
        emit(
            &Event::Phase {
                id: r.id.clone(),
                phase: Phase::Sampling,
            },
            &[],
        )?;
        for completed in 1..=p::STEPS {
            emit(
                &Event::Progress {
                    id: r.id.clone(),
                    completed,
                    total: p::STEPS,
                    elapsed_seconds: 0.,
                },
                &[],
            )?;
        }
        let frame = vec![0; r.shape.frame_bytes()];
        for index in 0..r.shape.frames {
            emit(
                &Event::VideoFrame {
                    id: r.id.clone(),
                    index,
                },
                &frame,
            )?;
            if index == 0 && r.prompt == "SYNTHETIC_PARTIAL_FAILURE" {
                return Err(ErrorCode::GenerationFailed);
            }
        }
        let audio = vec![0; r.shape.audio_samples() as usize * p::CHANNELS as usize * 4];
        emit(
            &Event::AudioChunk {
                id: r.id.clone(),
                start: 0,
                samples: r.shape.audio_samples(),
            },
            &audio,
        )
    }
}
fn run() -> Result<(), ErrorCode> {
    let root = PathBuf::from(std::env::var_os("H3_CPU_ROOT").ok_or(ErrorCode::DeploymentMismatch)?);
    let library = root.join("cpu-dependency.txt");
    let dependencies = BTreeMap::from([(
        library
            .to_str()
            .ok_or(ErrorCode::DeploymentMismatch)?
            .into(),
        deployment::sha(&library)?,
    )]);
    let executable = std::env::current_exe().map_err(|_| ErrorCode::DeploymentMismatch)?;
    let ready = deployment::identity(&executable, &dependencies, &[], &[])?;
    wire::serve(
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
        ready,
        |_| {
            std::fs::write(root.join("admitted"), b"own CPU fixture")
                .map_err(|_| ErrorCode::GenerationFailed)?;
            Ok(Fake)
        },
    )
}
fn main() {
    if !wire::quiet_entry(run) {
        std::process::exit(1)
    }
}
