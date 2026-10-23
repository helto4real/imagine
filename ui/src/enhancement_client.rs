use super::*;
use crate::enhancement_types::{ModelsRequest, Profile, ProfileUpdate, Request, Status};

impl Client {
    pub async fn enhancement_models(&self, lookup: &ModelsRequest) -> Result<Vec<String>, Failure> {
        serde_json::from_value(
            Self::json(
                self.post("/v1/enhancement/models")
                    .json(lookup)
                    .send()
                    .await,
            )
            .await?,
        )
        .map_err(|_| Failure::terminal("Invalid model list from server."))
    }
    pub async fn enhancement_profiles(&self) -> Result<Vec<Profile>, Failure> {
        serde_json::from_value(Self::json(self.get("/v1/enhancement/profiles").send().await).await?)
            .map_err(|_| Failure::terminal("Invalid enhancement profiles from server."))
    }
    pub async fn save_enhancement_profile(
        &self,
        update: ProfileUpdate,
    ) -> Result<Vec<Profile>, Failure> {
        serde_json::from_value(
            Self::json(
                self.http
                    .put(format!(
                        "{}/v1/enhancement/profiles/{}",
                        self.base, update.profile.id
                    ))
                    .json(&update)
                    .send()
                    .await,
            )
            .await?,
        )
        .map_err(|_| Failure::terminal("Invalid enhancement profiles from server."))
    }
    pub async fn delete_enhancement_profile(&self, id: &str) -> Result<Vec<Profile>, Failure> {
        serde_json::from_value(
            Self::json(
                self.http
                    .delete(format!("{}/v1/enhancement/profiles/{id}", self.base))
                    .send()
                    .await,
            )
            .await?,
        )
        .map_err(|_| Failure::terminal("Invalid enhancement profiles from server."))
    }
    pub async fn start_enhancement(&self, request: &Request) -> Result<(), Failure> {
        let response = self
            .post("/v1/enhancements")
            .json(request)
            .send()
            .await
            .map_err(|_| Failure::retry("Cannot reach Imagine. Reconnect and try again."))?;
        if response.status() != reqwest::StatusCode::ACCEPTED {
            Self::json(Ok(response)).await?;
            return Err(Failure::terminal(
                "Invalid enhancement response from server.",
            ));
        }
        Ok(())
    }
    pub async fn wait_enhancement(&self, id: &str) -> Result<String, Failure> {
        loop {
            let value =
                Self::json(self.get(&format!("/v1/enhancements/{id}")).send().await).await?;
            let status: Status = serde_json::from_value(value)
                .map_err(|_| Failure::terminal("Invalid enhancement result from server."))?;
            match status {
                Status::Running => tokio::time::sleep(Duration::from_millis(150)).await,
                Status::Succeeded { text } => return Ok(text),
                Status::Failed { code } => return Err(Failure::terminal(error_message(&code))),
                Status::Cancelled => return Err(Failure::terminal("Enhancement cancelled.")),
            }
        }
    }
    pub async fn remove_enhancement(&self, id: &str) {
        let _ = self
            .http
            .delete(format!("{}/v1/enhancements/{id}", self.base))
            .send()
            .await;
    }
}

pub(super) fn message(code: &str) -> Option<&'static str> {
    Some(match code {
        "invalid_enhancement_profile" => {
            "Enter a name, model ID, HTTP(S) base URL and system prompt. Check sampling parameters and reserved JSON fields."
        }
        "invalid_enhancement_url" => {
            "Enter an HTTP(S) base URL without credentials, query or fragment."
        }
        "enhancement_models_unsupported" => {
            "The server has no /models endpoint. Enter a model ID manually."
        }
        "enhancement_invalid_models" => {
            "The server returned an invalid model list. Check its OpenAI-compatible endpoint."
        }
        "enhancement_profile_not_found" => {
            "This enhancement profile no longer exists. Choose another in Settings."
        }
        "enhancement_profile_limit" => {
            "Remove an unused enhancement profile before adding another."
        }
        "enhancement_save_failed" => {
            "Could not save enhancement profiles. Check server-directory write permissions."
        }
        "invalid_enhancement_request" => "Enter a nonempty prompt of at most 64 KiB.",
        "enhancement_busy" => "Too many enhancements are active. Wait and try again.",
        "enhancement_timeout" => "The LLM timed out. Try again or reduce the output token limit.",
        "enhancement_unreachable" => {
            "Cannot reach the LLM. Check the profile's base URL and start the proxy."
        }
        "enhancement_auth_failed" => {
            "The LLM requires authentication. Use a local provider endpoint that does not require an API key."
        }
        "enhancement_rate_limited" => "The LLM is rate limited. Wait and try again.",
        "enhancement_request_rejected" => {
            "The LLM rejected the request. Check the base URL, model ID, token parameter and sampling settings."
        }
        "enhancement_provider_failed" => "The LLM failed to complete the request. Try again.",
        "enhancement_invalid_response" => {
            "The LLM returned an invalid response. Check that it supports Chat Completions with text output."
        }
        "enhancement_truncated" => {
            "The LLM reached the output token limit. Increase Max tokens; the prompt was not changed."
        }
        "enhancement_refused" => "The LLM declined the request. The prompt was not changed.",
        "enhancement_empty_response" => {
            "The LLM returned no prompt text. Check its system prompt and output token limit."
        }
        "enhancement_not_found" => "The enhancement expired or Imagine restarted. Try again.",
        "enhancement_id_exists" => "This enhancement request already exists. Try again.",
        _ => return None,
    })
}
