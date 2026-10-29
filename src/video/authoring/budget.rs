//! Director's outline budget: what an outline start would send to the
//! provider, measured without sending it.
use super::{engine::*, profiles, steps};
use crate::{enhancement::ProviderSnapshot, video::contract::VideoProject};
use serde_json::{Map, Value, json};
use sha2::Digest;
type Result<T> = std::result::Result<T, &'static str>;

/// Director's reserve beside the outline's output budget.
pub const CONTEXT_RESERVE_TOKENS: u64 = 8;

/// The outline's system and user messages and its resolved settings, as the
/// provider path builds them for `draft`.
pub(crate) struct Basis {
    pub system: String,
    pub user: String,
    pub parameters: Map<String, Value>,
}
pub(crate) fn basis(
    draft: &Draft,
    base: &VideoProject,
    profile: &profiles::Snapshot,
) -> Result<Basis> {
    let built = steps::outline_message(draft, base, steps::Notes::Known(&draft.inventory))?;
    Ok(Basis {
        system: system_for(
            profile,
            steps::outline_key(draft.request.add_dialogue),
            &built.contract,
        )?,
        parameters: profiles::resolve(
            &profile.profile.generation,
            &draft.request.provider,
            "story_outline",
        )?,
        user: built.user,
    })
}
/// The measurement of `basis` for a model with `context_tokens`; the
/// fingerprint covers the messages, provider, settings, profile and budget.
pub(crate) fn measure(
    basis: &Basis,
    draft: &Draft,
    provider: &ProviderSnapshot,
    profile: &profiles::Snapshot,
    context_tokens: u64,
) -> Result<Measurement> {
    let max_output_tokens = basis.parameters["max_tokens"]
        .as_u64()
        .ok_or("authoring_budget_unavailable")?;
    let text_bytes = basis.system.len() + basis.user.len();
    let budget = json!({"text_bytes":text_bytes,"message_count":2,"context_tokens":context_tokens,
        "max_output_tokens":max_output_tokens,"context_reserve_tokens":CONTEXT_RESERVE_TOKENS});
    let p = &provider.profile;
    let basis_json = json!({"system":basis.system,"user":basis.user,"provider":draft.request.provider,
        "provider_profile":{"id":p.id,"base_url":p.base_url,"model":p.model},"settings":basis.parameters,
        "prompt_profile":{"id":profile.resolved_id,"revision":profile.profile.revision},"budget":budget});
    Ok(Measurement {
        schema_version: 1,
        operation: Operation::Outline,
        chars: basis.user.chars().count(),
        char_limit: steps::CONTEXT_LIMIT,
        text_bytes,
        message_count: 2,
        context_tokens,
        max_output_tokens,
        context_reserve_tokens: CONTEXT_RESERVE_TOKENS,
        fingerprint: format!(
            "sha256:{:x}",
            sha2::Sha256::digest(basis_json.to_string().as_bytes())
        ),
    })
}
