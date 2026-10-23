//! Video shares admission and persistent destination checks with image jobs.
use super::*;
use crate::{h3_protocol as h3, short_av};
pub struct VideoCompleted {
    pub saved: std::path::PathBuf,
    pub av: short_av::Av,
}
pub fn video_request(prompt: &str) -> Value {
    video_request_with_selection(prompt, &h3::Selection::default()).expect("default H3 selection")
}
pub fn video_request_with_selection(
    prompt: &str,
    selection: &h3::Selection,
) -> Result<Value, &'static str> {
    request_for_profile(prompt, selection, None)
}
pub fn i2v_request(
    prompt: &str,
    selection: &h3::Selection,
    asset: &str,
    dimensions: (u32, u32),
) -> Result<Value, &'static str> {
    if asset.is_empty() {
        return Err("Choose a generated image with Animate first.");
    }
    request_for_profile(prompt, selection, Some((asset, dimensions)))
}
fn request_for_profile(
    prompt: &str,
    selection: &h3::Selection,
    source: Option<(&str, (u32, u32))>,
) -> Result<Value, &'static str> {
    let shape = selection.resolve_with_source(source.map(|(_, d)| d))?;
    let (operation, profile) = if source.is_some() {
        ("image_to_video", h3::Profile::ImageToAvV1)
    } else {
        ("text_to_video", h3::Profile::AvNoReferenceV2)
    };
    Ok(
        json!({"model":"minimax-h3","operation":operation,"prompt":prompt,
    "width":shape.width,"height":shape.height,"seed":42,
    "sampling":{"steps":h3::STEPS,"cfg_scale":1.,"sampler":h3::SAMPLER,"scheduler":h3::SCHEDULER},
    "output":{"format":"mp4","count":1},
    "inputs":{"start_image":source.map(|(id,_)|id)},
    "video":{"profile":profile,"frames":shape.frames,"fps":shape.fps,"audio":true},
    "client_metadata":{"h3_selection":selection}}),
    )
}

impl Client {
    pub async fn video_available(&self) -> std::result::Result<bool, Failure> {
        let metadata = Self::json(self.get("/v1/models/minimax-h3").send().await).await?;
        Ok(metadata["available"] == true
            && metadata["execution_enabled"] == true
            && metadata["profile"] == serde_json::to_value(h3::Profile::AvNoReferenceV2).unwrap())
    }
    pub async fn generate_video(
        &self,
        mut submission: Submission,
        mut progress: impl FnMut(Progress),
    ) -> std::result::Result<VideoCompleted, Failure> {
        let requested = h3::Shape {
            width: submission.request["width"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0),
            height: submission.request["height"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0),
            frames: submission.request["video"]["frames"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0),
            fps: submission.request["video"]["fps"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0),
        };
        if !requested.valid() {
            return Err(Failure::terminal("Invalid H3 dimensions or duration."));
        }
        self.prepare_destination(&mut submission).await?;
        let result = self.wait_for_job(&mut submission, &mut progress).await?;
        let saved = result["output"]["saved_path"]
            .as_str()
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_absolute() && p.starts_with(&submission.folder))
            .ok_or_else(|| {
                Failure::terminal("The server did not confirm persistent video storage.")
            })?;
        if result["output"]["persistent"] != true {
            return Err(Failure::terminal(
                "The server did not confirm persistent video storage.",
            ));
        }
        let assets = result["output"]["assets"]
            .as_array()
            .ok_or_else(|| Failure::retry("The video result is incomplete. Retry."))?;
        if assets.len() != 1
            || assets[0]["role"] != "video"
            || assets[0]["mime_type"] != short_av::MIME
        {
            return Err(Failure::retry("The video result is incomplete. Retry."));
        }
        let id = identifier(&assets[0]["id"])?;
        let mut response = self
            .get(&format!("/v1/assets/{id}"))
            .send()
            .await
            .map_err(|_| Failure::retry("Could not download the video. Retry."))?;
        if !response.status().is_success()
            || response
                .content_length()
                .is_some_and(|n| n > short_av::MAX_CONTAINER as u64)
        {
            return Err(Failure::retry(
                "Could not download the complete video. Retry.",
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Failure::retry("The video download was interrupted. Retry."))?
        {
            if bytes.len() + chunk.len() > short_av::MAX_CONTAINER {
                return Err(Failure::terminal("The video exceeds the supported size."));
            }
            bytes.extend_from_slice(&chunk);
        }
        let av = short_av::decode(&bytes)
            .map_err(|_| Failure::retry("Could not decode the video and audio. Retry."))?;
        if av.shape != requested {
            return Err(Failure::terminal(
                "Video dimensions or duration do not match the request.",
            ));
        }
        Ok(VideoCompleted { saved, av })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_locked_request_preserves_prompt_and_has_no_image_settings() {
        let value = video_request("  synthetic åäö\ntext  ");
        let r: imagine_server::contract::JobRequest =
            serde_json::from_value(value.clone()).unwrap();
        r.validate().unwrap();
        assert_eq!(r.prompt, "  synthetic åäö\ntext  ");
        assert!(value.get("model_options").is_none());
        assert_eq!(r.seed, Some(42));
        assert_eq!(r.sampling.steps, 20);
        assert_eq!(r.sampling.sampler, "res_multistep");
        assert_eq!(r.sampling.scheduler, "simple");
        assert_eq!(r.sampling.cfg_scale, 1.);
        assert_eq!(r.output.format, "mp4");
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    #[test]
    fn selections_pass_the_product_contract_and_keep_normalized_shape() {
        for ratio in h3::geometry::RATIOS {
            for quality in h3::geometry::QUALITIES {
                let selection = h3::Selection {
                    ratio: ratio.into(),
                    quality: quality.into(),
                    portrait: true,
                    seconds: 5.,
                };
                let shape = selection.resolve().unwrap();
                let value = video_request_with_selection("synthetic", &selection).unwrap();
                let r: imagine_server::contract::JobRequest =
                    serde_json::from_value(value).unwrap();
                r.validate().unwrap();
                assert_eq!(r.sampling.steps, 20);
                assert_eq!(r.sampling.sampler, "res_multistep");
                assert_eq!(
                    serde_json::to_value(imagine_server::video::job::request(&r).unwrap().shape)
                        .unwrap(),
                    serde_json::to_value(shape).unwrap()
                );
            }
        }
    }
}

#[cfg(test)]
mod i2v_tests {
    use super::*;
    #[test]
    fn start_image_request_uses_source_geometry_and_locked_profile() {
        let selection = h3::Selection {
            ratio: "Source".into(),
            ..Default::default()
        };
        let value =
            i2v_request("synthetic motion", &selection, "asset_owned", (1000, 1500)).unwrap();
        let r: imagine_server::contract::JobRequest =
            serde_json::from_value(value.clone()).unwrap();
        r.validate().unwrap();
        assert_eq!(value["inputs"]["start_image"], "asset_owned");
        assert_eq!(value["video"]["profile"], "h3-fl2va-i2va-v1");
        assert!(r.height > r.width);
        assert!(video_request_with_selection("text", &selection).is_err());
        assert!(i2v_request("motion", &selection, "", (1000, 1500)).is_err());
    }
}
