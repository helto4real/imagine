//! CPU schedule/handoff evidence, without Runtime, checkpoints or CUDA initialization.
use imagine_inference::{
    h3_product::{
        protocol::{self as p, sampling::*},
        sampling,
    },
    native::Error,
};
use serde_json::json;
use tch::Tensor;
fn values(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(tensor).unwrap()
}
fn main() -> Result<(), Error> {
    tch::set_num_threads(1);
    let first = Stage {
        sigmas: vec![1., 0.9, 0.6],
        ..Stage::default()
    };
    let second = SecondStage {
        sampling: Stage {
            steps: 7,
            ..Stage::default()
        },
        profile: p::Profile::AvNoReferenceV2,
        handoff: Handoff::ContinueLatent,
        upscale: None,
        model_file: None,
    };
    let spec = Sampling {
        first,
        second: Some(second),
        ..Sampling::default()
    };
    let mut cases = vec![];
    for curve in [
        vec![1f32, 0.8, 0.6, 0.4, 0.2, 0.],
        vec![0.8, 0.8, 0.5, 0.5, 0.2, 0.],
        vec![0.4, 0.2, 0.],
    ] {
        let schedules =
            sampling::resolve(&spec, |_| Tensor::f_from_slice(&curve).map_err(Error::from))?;
        cases.push(
            json!({"curve":curve,"first":values(&schedules[0]),"second":values(&schedules[1])}),
        );
    }
    let prediction = Tensor::f_from_slice(&[-0.2f32, 0.7, 0.3])?;
    let noise = Tensor::f_from_slice(&[0.4f32, -0.1, 0.8])?;
    let renoised = sampling::renoise_video(&prediction, &noise, 0.6f32.into())?;
    println!(
        "{}",
        json!({"device":"cpu","total_steps":spec.total_steps(),"cases":cases,"prediction":values(&prediction),"noise":values(&noise),"boundary":0.6f32,"renoised":values(&renoised)})
    );
    Ok(())
}
