//! Video shares admission/poll/retry with image jobs; Save is an explicit UI action.
use super::*;
use crate::{h3_protocol as h3, short_av};
pub struct VideoCompleted {
    pub bytes: std::sync::Arc<Vec<u8>>,
    pub av: short_av::Av,
}
pub fn video_request(prompt: &str) -> Value {
    json!({"model":"minimax-h3","operation":"text_to_video","prompt":prompt,
    "width":h3::WIDTH,"height":h3::HEIGHT,"seed":42,
    "sampling":{"steps":h3::STEPS,"cfg_scale":1.,"sampler":"euler","scheduler":"simple"},
    "output":{"format":"mkv","count":1},
    "video":{"profile":h3::Profile::ShortAvNoReferenceV1,"frames":h3::FRAMES,"fps":h3::FPS,"audio":true}})
}
impl Client {
    pub async fn video_available(&self) -> std::result::Result<bool, Failure> {
        let metadata = Self::json(self.get("/v1/models/minimax-h3").send().await).await?;
        Ok(metadata["available"] == true
            && metadata["execution_enabled"] == true
            && metadata["profile"]
                == serde_json::to_value(h3::Profile::ShortAvNoReferenceV1).unwrap())
    }
    pub async fn generate_video(
        &self,
        mut submission: Submission,
        mut progress: impl FnMut(Progress),
    ) -> std::result::Result<VideoCompleted, Failure> {
        let result = self.wait_for_job(&mut submission, &mut progress).await?;
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
        Ok(VideoCompleted {
            bytes: std::sync::Arc::new(bytes),
            av,
        })
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
        assert_eq!(r.output.format, "mkv");
    }
}
