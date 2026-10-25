//! Project snapshot responses as read by UI clients. The server's persistence
//! types stay authoritative; these views ignore fields a client does not read.
use crate::{
    assembly::Plan,
    captures::Capture,
    contract::{AssetId, SceneId, TakeId, VideoProject},
    review::TakeState,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaveState {
    SessionOnly,
    Saving,
    Saved,
    NotSaved,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistenceStatus {
    pub state: SaveState,
    pub saved_revision: Option<u64>,
    pub error_code: Option<String>,
    /// The folder a bound project saves to; absent while session-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<std::path::PathBuf>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TakeStatus {
    Candidate,
    Accepted,
    Rejected,
    Outdated,
}
impl TakeStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Outdated => "outdated",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TakeView {
    pub id: TakeId,
    pub scene_id: SceneId,
    pub status: TakeStatus,
    /// Generation recipe, shown as metadata; None for an imported video.
    pub provenance: Option<Value>,
}
impl TakeView {
    pub fn source_take(&self) -> Option<&str> {
        self.provenance.as_ref()?["source_take"].as_str()
    }
}
#[derive(Debug, Clone, Deserialize)]
pub struct ResourceView {
    #[serde(default)]
    pub id: String,
    /// Tagged by `kind`, for example `h3_endpoint` with its `take_id`.
    pub kind: Value,
    #[serde(default)]
    pub sha256: String,
}
impl ResourceView {
    pub fn kind(&self) -> &str {
        self.kind["kind"].as_str().unwrap_or("")
    }
    pub fn take_id(&self) -> Option<&str> {
        self.kind["take_id"].as_str()
    }
    pub fn asset_id(&self) -> Option<&str> {
        self.kind["asset_id"].as_str()
    }
}
#[derive(Debug, Clone, Deserialize)]
pub struct DeliveryView {
    pub id: String,
    pub asset_id: AssetId,
    pub plan: Plan,
}

/// A project snapshot response, without its project.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SnapshotView {
    pub persistence: Option<PersistenceStatus>,
    pub runtime_assets: BTreeMap<AssetId, String>,
    pub runtime_takes: BTreeMap<TakeId, String>,
    pub take_states: BTreeMap<TakeId, TakeState>,
    pub takes: Vec<TakeView>,
    pub captures: BTreeMap<TakeId, Capture>,
    pub deliveries: Vec<DeliveryView>,
    pub resources: Vec<ResourceView>,
    pub storage_layout: Option<u32>,
    pub authoring: Vec<crate::authoring::Draft>,
    pub preset_states: BTreeMap<SceneId, crate::creative::PresetState>,
}
/// A full snapshot response with its validated project.
pub fn parse(raw: &Value) -> Option<(VideoProject, SnapshotView)> {
    let project: VideoProject = serde_json::from_value(raw["project"].clone()).ok()?;
    project.validate().ok()?;
    Some((project, serde_json::from_value(raw.clone()).ok()?))
}
