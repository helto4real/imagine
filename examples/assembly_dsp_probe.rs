//! Opt-in CPU numeric probe. Inputs/outputs are own synthetic test data, not runtime jobs.
use imagine_server::video::assembly::{self, dsp};
use serde_json::{Value, json};
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let input = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("input required"))?,
    );
    let output = PathBuf::from(
        std::env::args_os()
            .nth(2)
            .ok_or_else(|| anyhow::anyhow!("output required"))?,
    );
    for path in [&input, &output] {
        anyhow::ensure!(
            path.starts_with("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e"),
            "own encrypted probe paths required"
        );
    }
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(input)?)?;
    let mut results = vec![];
    for case in cases {
        let op = case["op"].as_str().unwrap_or("");
        let samples = || serde_json::from_value::<Vec<dsp::Stereo>>(case["source"].clone());
        let value = match op {
            "resize" => {
                let source: Vec<u8> = serde_json::from_value(case["source"].clone())?;
                json!(assembly::fit_rgb_frame(
                    &source,
                    case["width"].as_u64().unwrap() as u32,
                    case["height"].as_u64().unwrap() as u32,
                    &imagine_server::video::contract::DeliverySettings {
                        width: case["target_width"].as_u64().unwrap() as u32,
                        height: case["target_height"].as_u64().unwrap() as u32,
                        fps: imagine_server::video::contract::Time { num: 24, den: 1 }
                    }
                ))
            }
            "frame_fit" => {
                let source = case["source_count"].as_u64().unwrap();
                let count = case["count"].as_u64().unwrap();
                json!(
                    (0..count)
                        .map(|i| assembly::fit_index(i, source, count))
                        .collect::<Vec<_>>()
                )
            }
            "audio_fit" => json!(dsp::fit_audio(
                &samples()?,
                case["count"].as_u64().unwrap() as usize
            )),
            "gain_fades" => {
                let mut samples = samples()?;
                dsp::gain_fades(
                    &mut samples,
                    case["gain"].as_f64().unwrap() as f32,
                    case["fade_in"].as_u64().unwrap() as usize,
                    case["fade_out"].as_u64().unwrap() as usize,
                );
                json!(samples)
            }
            "normalize" => {
                let mut samples = samples()?;
                dsp::normalize(&mut samples);
                json!(samples)
            }
            "combine" => {
                let mut samples = samples()?;
                let timeline =
                    serde_json::from_value::<Vec<dsp::Stereo>>(case["timeline"].clone())?;
                let options = serde_json::from_value(case["options"].clone())?;
                dsp::combine(&mut samples, &timeline, &options);
                json!(samples)
            }
            "tone" => {
                let left: Vec<Vec<f32>> = serde_json::from_value(case["left"].clone())?;
                let mut right = serde_json::from_value::<Vec<Vec<f32>>>(case["right"].clone())?;
                dsp::tone_match(&left, &mut right, case["fps"].as_f64().unwrap());
                json!(right)
            }
            "blend" => {
                let mut left: Vec<Vec<f32>> = serde_json::from_value(case["left"].clone())?;
                let right: Vec<Vec<f32>> = serde_json::from_value(case["right"].clone())?;
                let count = left.len() as u64;
                for (i, (a, b)) in left.iter_mut().zip(right).enumerate() {
                    dsp::blend(a, &b, i as u64, count);
                }
                json!(left)
            }
            _ => anyhow::bail!("unknown probe operation"),
        };
        results.push(json!({"id":case["id"],"value":value}));
    }
    std::fs::write(output, serde_json::to_vec(&results)?)?;
    Ok(())
}
