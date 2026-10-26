//! Explicit project persistence. All filesystem access starts at caller-selected roots.
use super::{
    bundle::{self, Bundle, Directory, Store},
    contract::*,
    editing::{self, EditRequest},
};
pub use imagine_video_contract::snapshot::{PersistenceStatus, SaveState, TakeStatus};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageError {
    pub code: String,
}
impl StorageError {
    pub(crate) fn new(code: &str) -> Self {
        Self { code: code.into() }
    }
}
impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.code)
    }
}
impl std::error::Error for StorageError {}
impl From<Issue> for StorageError {
    fn from(issue: Issue) -> Self {
        Self::new(&issue.code)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bucket {
    Media,
    Takes,
    Continuity,
}
impl Bucket {
    pub(crate) fn directory(&self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Takes => "takes",
            Self::Continuity => "continuity",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlobKey {
    pub bucket: Bucket,
    pub sha256: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResourceKind {
    Reference {
        asset_id: AssetId,
        revision: u64,
    },
    TakeVideo {
        take_id: TakeId,
    },
    TakeAudio {
        take_id: TakeId,
    },
    H3Endpoint {
        take_id: TakeId,
        schema_version: u32,
        model: ModelSelection,
        shape: crate::h3_protocol::Shape,
    },
    H3Motion {
        take_id: TakeId,
        model: ModelSelection,
        descriptor: crate::h3_protocol::continuation::Descriptor,
    },
    /// Opaque data, but version and compatibility identity are explicit; runtime verifies contents.
    Continuation {
        take_id: TakeId,
        format: String,
        format_version: u32,
        model: ModelSelection,
        width: u32,
        height: u32,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resource {
    pub id: String,
    pub kind: ResourceKind,
    pub sha256: String,
    pub bytes: u64,
    /// User-selected original. None retains the legacy owned-blob representation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<PathBuf>,
}
impl Resource {
    pub fn key(&self) -> BlobKey {
        BlobKey {
            bucket: match self.kind {
                ResourceKind::Reference { .. } => Bucket::Media,
                ResourceKind::TakeVideo { .. } | ResourceKind::TakeAudio { .. } => Bucket::Takes,
                ResourceKind::Continuation { .. }
                | ResourceKind::H3Endpoint { .. }
                | ResourceKind::H3Motion { .. } => Bucket::Continuity,
            },
            sha256: self.sha256.clone(),
        }
    }
    pub fn inspect(id: String, kind: ResourceKind, file: &mut File) -> Result<Self, StorageError> {
        let (sha256, bytes) = bundle::digest(file)?;
        let resource = Self {
            id,
            kind,
            sha256,
            bytes,
            source: None,
        };
        resource.validate()?;
        Ok(resource)
    }
    /// Describe an original without copying its bytes into session/project storage.
    pub fn external(
        id: String,
        kind: ResourceKind,
        path: &Path,
        file: &mut File,
    ) -> Result<Self, StorageError> {
        if !path.is_absolute() {
            return Err(StorageError::new("invalid_source_path"));
        }
        let original = open_original(path)?;
        let before = source_witness(file)?;
        if before != source_witness(&original)? {
            return Err(StorageError::new("source_changed"));
        }
        let mut resource = Self::inspect(id, kind, file)?;
        if before != source_witness(file)? {
            return Err(StorageError::new("source_changed"));
        }
        resource.source = Some(path.to_owned());
        resource.validate()?;
        Ok(resource)
    }
    /// Open the selected original with an independent position. The caller owns
    /// this descriptor for decoding; no source bytes are staged or rewritten.
    pub fn open_source(&self) -> Result<Option<File>, StorageError> {
        self.source
            .as_deref()
            .map(|path| open_external(path, &self.sha256, self.bytes))
            .transpose()
    }
    fn validate(&self) -> Result<(), StorageError> {
        if self.source.as_ref().is_some_and(|path| !path.is_absolute()) {
            return Err(StorageError::new("invalid_source_path"));
        }
        if self.id.trim().is_empty()
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
            || self.bytes > MAX_INTEGER
        {
            return Err(StorageError::new("invalid_resource"));
        }
        match &self.kind {
            ResourceKind::H3Motion { descriptor, .. }
                if !descriptor.valid() || self.bytes != descriptor.bytes() as u64 =>
            {
                return Err(StorageError::new("invalid_continuation"));
            }
            ResourceKind::H3Endpoint {
                schema_version,
                shape,
                ..
            } if *schema_version == 0 || !shape.valid() => {
                return Err(StorageError::new("invalid_endpoint"));
            }
            ResourceKind::Reference { asset_id, revision }
                if asset_id.0.trim().is_empty() || *revision > MAX_INTEGER =>
            {
                return Err(StorageError::new("invalid_reference"));
            }
            ResourceKind::Continuation {
                format,
                format_version,
                width,
                height,
                ..
            } if format.trim().is_empty()
                || *format_version == 0
                || *width == 0
                || *height == 0 =>
            {
                return Err(StorageError::new("invalid_continuation"));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Open a linked original with its stored content identity, without staging or
/// rewriting the source. The returned descriptor has an independent position.
pub fn open_external(path: &Path, sha256: &str, bytes: u64) -> Result<File, StorageError> {
    let mut file = open_original(path)?;
    let before = source_witness(&file)?;
    let actual = bundle::digest(&mut file).map_err(|_| StorageError::new("source_unavailable"))?;
    if before != source_witness(&file)? || actual.0 != sha256 || actual.1 != bytes {
        return Err(StorageError::new("source_changed"));
    }
    Ok(file)
}

fn open_original(path: &Path) -> Result<File, StorageError> {
    use std::os::unix::fs::OpenOptionsExt;
    if !path.is_absolute() {
        return Err(StorageError::new("invalid_source_path"));
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| {
            StorageError::new(match error.kind() {
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory => {
                    "source_missing"
                }
                _ => "source_unavailable",
            })
        })?;
    source_witness(&file)?;
    Ok(file)
}

#[derive(PartialEq, Eq)]
pub(crate) struct SourceWitness {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
pub(crate) fn source_witness(file: &File) -> Result<SourceWitness, StorageError> {
    use std::os::unix::fs::MetadataExt;
    let value = file
        .metadata()
        .map_err(|_| StorageError::new("source_unavailable"))?;
    if !value.is_file() {
        return Err(StorageError::new("invalid_source_file"));
    }
    Ok(SourceWitness {
        device: value.dev(),
        inode: value.ino(),
        bytes: value.len(),
        modified: (value.mtime(), value.mtime_nsec()),
        changed: (value.ctime(), value.ctime_nsec()),
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetVersion {
    pub asset_id: AssetId,
    pub revision: u64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TakeProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_loras: Option<imagine_video_contract::loras::Stack>,
    pub model: ModelSelection,
    pub seed: u64,
    pub input_revision: String,
    pub compiled_prompt: String,
    pub settings: SettingsSlot,
    pub references: Vec<AssetVersion>,
    pub source_take: Option<TakeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuity: Option<EndpointSource>,
    /// The Scene's prompt text when the Take was made, so selecting it can restore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored: Option<AuthoredPrompt>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPrompt {
    pub description: String,
    pub sections: Vec<AuthoredSection>,
    pub summary: String,
    pub ambient_sound: String,
    pub non_diegetic_music: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSection {
    pub id: SectionId,
    pub prompt: String,
}
impl AuthoredPrompt {
    pub fn of(project: &VideoProject, scene: &Scene) -> Self {
        let prompt = project.creative.scene_prompts.get(&scene.id);
        let text = |f: fn(&imagine_video_contract::creative::ScenePrompt) -> &String| {
            prompt.map(f).cloned().unwrap_or_default()
        };
        Self {
            description: scene.description.clone(),
            sections: scene
                .sections
                .iter()
                .map(|s| AuthoredSection {
                    id: s.id.clone(),
                    prompt: s.prompt.clone(),
                })
                .collect(),
            summary: text(|p| &p.summary),
            ambient_sound: text(|p| &p.ambient_sound),
            non_diegetic_music: text(|p| &p.non_diegetic_music),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointSource {
    pub scene_id: SceneId,
    pub take_id: TakeId,
    pub resource_id: String,
    pub sha256: String,
    pub schema_version: u32,
    pub shape: crate::h3_protocol::Shape,
    pub method: ContinuityMethod,
    #[serde(default)]
    pub include_audio: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredTake {
    pub id: TakeId,
    pub scene_id: SceneId,
    pub status: TakeStatus,
    /// None is an imported video with no known generation recipe, not a default recipe.
    pub provenance: Option<TakeProvenance>,
}
impl StoredTake {
    pub fn generated(&self) -> Result<&TakeProvenance, StorageError> {
        self.provenance
            .as_ref()
            .ok_or_else(|| StorageError::new("take_generation_recipe_unknown"))
    }
    pub fn accepted_summary(&self, resources: &[Resource]) -> Result<AcceptedTake, StorageError> {
        let (model, source_fingerprint) = if let Some(provenance) = &self.provenance {
            (
                Some(provenance.model.clone()),
                provenance.input_revision.clone(),
            )
        } else {
            let video = resources.iter().find(|r| matches!(&r.kind, ResourceKind::TakeVideo { take_id } if take_id == &self.id))
                .ok_or_else(|| StorageError::new("take_resource_missing"))?;
            (None, video.sha256.clone())
        };
        Ok(AcceptedTake {
            id: self.id.clone(),
            model,
            source_fingerprint,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryOrigin {
    pub library_id: String,
    pub entry_id: String,
    pub entry_revision: u64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectDocument {
    pub schema_version: u32,
    /// Only explicitly saved private projects use the new output/work layout.
    /// Absence retains legacy media/takes/continuity paths without migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_layout: Option<u32>,
    pub project: VideoProject,
    pub resources: Vec<Resource>,
    pub takes: Vec<StoredTake>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub captures: BTreeMap<TakeId, imagine_video_contract::captures::Capture>,
    pub origin: Option<LibraryOrigin>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub locked_takes: BTreeMap<SceneId, TakeId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authoring: Vec<imagine_video_contract::authoring::Draft>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deliveries: Vec<StoredDelivery>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredDelivery {
    pub id: String,
    pub asset_id: AssetId,
    pub plan: imagine_video_contract::assembly::Plan,
}
impl ProjectDocument {
    pub fn new(project: VideoProject) -> Self {
        Self {
            schema_version: 1,
            storage_layout: None,
            project,
            resources: vec![],
            takes: vec![],
            captures: BTreeMap::new(),
            origin: None,
            locked_takes: BTreeMap::new(),
            authoring: vec![],
            deliveries: vec![],
        }
    }
    /// The validated document after an edit, with changed continuity invalidated.
    pub(crate) fn edited(&self, request: EditRequest) -> Result<ProjectDocument, StorageError> {
        let mut candidate = self.clone();
        editing::apply(&mut candidate.project, request)?;
        super::review::invalidate_changed(&self.project, &mut candidate)?;
        candidate.validate()?;
        Ok(candidate)
    }
}
pub(crate) fn validate_resources(resources: &[Resource]) -> Result<(), StorageError> {
    let mut ids = BTreeSet::new();
    let mut sizes = BTreeMap::new();
    for resource in resources {
        resource.validate()?;
        if !ids.insert(&resource.id) {
            return Err(StorageError::new("duplicate_resource"));
        }
        if sizes
            .insert(resource.key(), resource.bytes)
            .is_some_and(|size| size != resource.bytes)
        {
            return Err(StorageError::new("invalid_resource"));
        }
    }
    Ok(())
}
impl Bundle for ProjectDocument {
    fn identity(&self) -> &str {
        &self.project.id.0
    }
    fn revision(&self) -> u64 {
        self.project.project_revision
    }
    fn resources(&self) -> Vec<&Resource> {
        self.resources.iter().collect()
    }
    fn resource_directory(&self, resource: &Resource) -> &'static str {
        if self.storage_layout != Some(1) {
            return resource.key().bucket.directory();
        }
        match &resource.kind {
            ResourceKind::TakeVideo { .. } => "outputs/videos",
            ResourceKind::TakeAudio { .. } => "outputs/audio",
            ResourceKind::Reference { asset_id, .. } => {
                match self.project.assets.get(asset_id).map(|asset| &asset.kind) {
                    Some(MediaKind::Video) => "outputs/videos",
                    Some(MediaKind::Audio) => "outputs/audio",
                    _ => "outputs/images",
                }
            }
            _ => "work/continuity",
        }
    }
    fn released_directories(&self, key: &BlobKey) -> Vec<&'static str> {
        if self.storage_layout != Some(1) {
            return vec![key.bucket.directory()];
        }
        match key.bucket {
            Bucket::Media => vec!["outputs/images", "outputs/videos", "outputs/audio"],
            Bucket::Takes => vec!["outputs/videos", "outputs/audio"],
            Bucket::Continuity => vec!["work/continuity"],
        }
    }
    fn prepared_directories(&self) -> &'static [&'static str] {
        if self.storage_layout == Some(1) {
            &["outputs", "work", "tmp", "autosave"]
        } else {
            &[]
        }
    }
    fn validate(&self) -> Result<(), StorageError> {
        if self.schema_version != 1 || self.storage_layout.is_some_and(|version| version != 1) {
            return Err(StorageError::new("unsupported_storage_schema"));
        }
        self.project.validate()?;
        for (id, capture) in &self.captures {
            if !self.takes.iter().any(|take| &take.id == id)
                || capture.source_capture.as_ref().is_some_and(|source| {
                    source == id || !self.takes.iter().any(|take| &take.id == source)
                })
            {
                return Err(StorageError::new("capture_source_missing"));
            }
            if let Some(source) = &capture.imported_from
                && !self.resources.iter().any(|r| matches!(&r.kind,ResourceKind::Reference {asset_id,revision} if asset_id == &source.asset_id && *revision == source.revision))
            { return Err(StorageError::new("capture_source_missing")); }
        }
        for (scene, id) in &self.locked_takes {
            if !self.takes.iter().any(|take| {
                &take.id == id
                    && &take.scene_id == scene
                    && matches!(take.status, TakeStatus::Accepted | TakeStatus::Outdated)
            }) {
                return Err(StorageError::new("locked_take_mismatch"));
            }
        }
        validate_resources(&self.resources)?;
        let reference = |asset: &AssetId, revision: u64| {
            self.resources.iter().any(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,revision:v} if asset_id==asset&&*v==revision))
        };
        for asset in self.project.assets.values() {
            if !reference(&asset.id, asset.revision) {
                return Err(StorageError::new("reference_resource_missing"));
            }
        }
        let mut delivery_ids = BTreeSet::new();
        for delivery in &self.deliveries {
            if !delivery_ids.insert(&delivery.id)
                || self
                    .project
                    .assets
                    .get(&delivery.asset_id)
                    .is_none_or(|a| a.kind != MediaKind::Video)
            {
                return Err(StorageError::new("invalid_delivery_resource"));
            }
        }
        let mut take_ids = BTreeSet::new();
        for take in &self.takes {
            if take.id.0.trim().is_empty()
                || !take_ids.insert(&take.id)
                // Completed outputs retain their original scene identity after a
                // scene is removed or Story replaces the timeline. Their frozen
                // prompt/settings, resource and provenance checks remain below.
                || take.scene_id.0.trim().is_empty()
                || take.provenance.as_ref().is_some_and(|p| p.model != p.settings.model)
            {
                return Err(StorageError::new("invalid_take"));
            }
            if !self
                .resources
                .iter()
                .any(|r| matches!(&r.kind,ResourceKind::TakeVideo{take_id} if take_id==&take.id))
            {
                return Err(StorageError::new("take_resource_missing"));
            }
            if let Some(provenance) = &take.provenance {
                for asset in &provenance.references {
                    if !reference(&asset.asset_id, asset.revision) {
                        return Err(StorageError::new("historical_reference_missing"));
                    }
                }
                if let Some(source) = &provenance.source_take
                    && !self.takes.iter().any(|t| &t.id == source)
                {
                    return Err(StorageError::new("source_take_missing"));
                }
                if let Some(source) = &provenance.continuity
                && (provenance.source_take.as_ref() != Some(&source.take_id)
                    || !self.takes.iter().any(|take| take.id == source.take_id && take.scene_id == source.scene_id)
                    || !self.resources.iter().any(|resource| resource.id == source.resource_id && resource.sha256 == source.sha256 && (matches!(&resource.kind, ResourceKind::H3Endpoint { take_id, schema_version, shape, .. } if source.method==ContinuityMethod::LastFrameFirstFrame && take_id == &source.take_id && *schema_version == source.schema_version && shape == &source.shape) || matches!(&resource.kind, ResourceKind::H3Motion {take_id,descriptor,..} if matches!(source.method,ContinuityMethod::MotionContext|ContinuityMethod::MotionContextExp) && take_id==&source.take_id && descriptor.schema_version==source.schema_version && descriptor.shape==source.shape))))
                { return Err(StorageError::new("continuity_source_mismatch")); }
            }
        }
        for scene in &self.project.scenes {
            let accepted: Vec<_> = self
                .takes
                .iter()
                .filter(|t| t.scene_id == scene.id && t.status == TakeStatus::Accepted)
                .collect();
            match &scene.accepted_take {
                Some(summary) => {
                    let selected = self
                        .takes
                        .iter()
                        .find(|t| t.id == summary.id && t.scene_id == scene.id)
                        .ok_or_else(|| StorageError::new("accepted_take_mismatch"))?;
                    if !matches!(selected.status, TakeStatus::Accepted | TakeStatus::Outdated)
                        || selected.accepted_summary(&self.resources)? != *summary
                        || accepted.iter().any(|t| t.id != summary.id)
                    {
                        return Err(StorageError::new("accepted_take_mismatch"));
                    }
                }
                None if accepted.is_empty() => {}
                None => return Err(StorageError::new("accepted_take_mismatch")),
            }
        }
        for resource in &self.resources {
            let take_id = match &resource.kind {
                ResourceKind::Reference { .. } => continue,
                ResourceKind::TakeVideo { take_id }
                | ResourceKind::TakeAudio { take_id }
                | ResourceKind::H3Endpoint { take_id, .. }
                | ResourceKind::H3Motion { take_id, .. }
                | ResourceKind::Continuation { take_id, .. } => take_id,
            };
            if !take_ids.contains(take_id) {
                return Err(StorageError::new("resource_take_missing"));
            }
            if let ResourceKind::H3Endpoint { model, take_id, .. }
            | ResourceKind::H3Motion { model, take_id, .. } = &resource.kind
                && !self.takes.iter().any(|take| {
                    &take.id == take_id
                        && take.provenance.as_ref().is_some_and(|p| &p.model == model)
                })
            {
                return Err(StorageError::new("endpoint_model_mismatch"));
            }
        }
        Ok(())
    }
}
/// Located private projects start in a new folder; only prepared, empty
/// project classes from an interrupted first save may already exist.
fn empty_project_folder(path: &Path) -> Result<(), StorageError> {
    let io = |_| StorageError::new("storage_io_failed");
    for entry in std::fs::read_dir(path).map_err(io)? {
        let entry = entry.map_err(io)?;
        let prepared = ["outputs", "work", "tmp", "autosave"]
            .iter()
            .any(|name| entry.file_name() == *name);
        if !prepared || !only_directories(&entry)? {
            return Err(StorageError::new("storage_not_empty"));
        }
    }
    Ok(())
}
/// A real directory holding nothing but empty directories, at any depth.
fn only_directories(entry: &std::fs::DirEntry) -> Result<bool, StorageError> {
    let io = |_| StorageError::new("storage_io_failed");
    if !entry.file_type().map_err(io)?.is_dir() {
        return Ok(false);
    }
    for child in std::fs::read_dir(entry.path()).map_err(io)? {
        if !only_directories(&child.map_err(io)?)? {
            return Ok(false);
        }
    }
    Ok(true)
}
/// Caller supplies an already-existing session directory. Staged copies are immediately
/// unlinked and live only through owned handles; no recovery files or hidden history.
pub struct SessionStaging {
    root: Directory,
}
impl SessionStaging {
    pub fn select(path: &Path) -> Result<Self, StorageError> {
        Ok(Self {
            root: Directory::select(path)?,
        })
    }
    pub(crate) fn copy(&self, source: &mut File) -> Result<File, StorageError> {
        let mut target = self.root.temporary()?;
        bundle::copy(source, &mut target)?;
        Ok(target)
    }
    pub(crate) fn temporary(&self) -> Result<File, StorageError> {
        self.root.temporary()
    }
}
/// All mutation/save methods require exclusive access. No background task is launched.
/// A caller may debounce edits; update() autosaves synchronously once Save has bound a root.
pub struct ProjectSession {
    document: ProjectDocument,
    sources: BTreeMap<BlobKey, File>,
    store: Option<Store<ProjectDocument>>,
    status: PersistenceStatus,
}
pub(crate) struct PreparedUpdate {
    document: ProjectDocument,
    sources: BTreeMap<BlobKey, File>,
}
impl ProjectSession {
    pub fn new(project: VideoProject) -> Result<Self, StorageError> {
        project.validate()?;
        Ok(Self {
            document: ProjectDocument::new(project),
            sources: BTreeMap::new(),
            store: None,
            status: PersistenceStatus {
                state: SaveState::SessionOnly,
                saved_revision: None,
                error_code: None,
                location: None,
            },
        })
    }
    pub fn document(&self) -> &ProjectDocument {
        &self.document
    }
    pub(crate) fn is_at(&self, path: &Path) -> Result<bool, StorageError> {
        self.store
            .as_ref()
            .map_or(Ok(false), |store| store.is_at(path))
    }
    /// Stream a verified resource to a caller-owned destination; no internal path escapes.
    pub fn copy_resource(
        &self,
        id: &str,
        target: &mut impl std::io::Write,
    ) -> Result<(), StorageError> {
        let source = self.resource_file(id)?;
        bundle::stream(&source, target)?;
        Ok(())
    }
    pub fn status(&self) -> &PersistenceStatus {
        &self.status
    }
    pub(crate) fn is_bound(&self) -> bool {
        self.store.is_some()
    }
    fn ensure_writable(&self) -> Result<(), StorageError> {
        self.store.as_ref().map_or(Ok(()), Store::check_writable)
    }
    pub fn resource_file(&self, id: &str) -> Result<File, StorageError> {
        use std::os::fd::AsRawFd;
        let resource = self
            .document
            .resources
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        if let Some(file) = resource.open_source()? {
            return Ok(file);
        }
        let source = self
            .sources
            .get(&resource.key())
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        let mut file = File::open(format!("/proc/self/fd/{}", source.as_raw_fd()))
            .map_err(|_| StorageError::new("storage_io_failed"))?;
        if bundle::digest(&mut file)? != (resource.sha256.clone(), resource.bytes) {
            return Err(StorageError::new("resource_integrity_failed"));
        }
        Ok(file)
    }
    /// Stage independent copies before changing the document. Existing sources are retained
    /// only while referenced by the new document, including historical Take references.
    pub fn update(
        &mut self,
        expected_revision: u64,
        document: ProjectDocument,
        inputs: BTreeMap<BlobKey, File>,
        staging: &SessionStaging,
    ) -> Result<(), StorageError> {
        let prepared = self.prepare_update(expected_revision, document, inputs, staging)?;
        self.commit_prepared(prepared);
        self.autosave()
    }
    pub(crate) fn prepare_update(
        &self,
        expected_revision: u64,
        mut document: ProjectDocument,
        inputs: BTreeMap<BlobKey, File>,
        staging: &SessionStaging,
    ) -> Result<PreparedUpdate, StorageError> {
        self.ensure_writable()?;
        if expected_revision != self.document.project.project_revision {
            return Err(StorageError::new("revision_conflict"));
        }
        if document.project.id != self.document.project.id {
            return Err(StorageError::new("project_identity_changed"));
        }
        document.project.project_revision = expected_revision
            .checked_add(1)
            .filter(|n| *n <= MAX_INTEGER)
            .ok_or_else(|| StorageError::new("revision_exhausted"))?;
        document.validate()?;
        let mut sources = bundle::clone_files(&self.sources)?;
        let owned: BTreeSet<_> = document
            .resources
            .iter()
            .filter(|r| r.source.is_none())
            .map(Resource::key)
            .collect();
        for (key, mut file) in inputs {
            if owned.contains(&key) {
                sources.insert(key, staging.copy(&mut file)?);
            }
        }
        for resource in &document.resources {
            if resource.source.is_some() {
                continue;
            }
            let file = sources
                .get_mut(&resource.key())
                .ok_or_else(|| StorageError::new("resource_missing"))?;
            if bundle::digest(file)? != (resource.sha256.clone(), resource.bytes) {
                return Err(StorageError::new("resource_integrity_failed"));
            }
        }
        sources.retain(|key, _| owned.contains(key));
        Ok(PreparedUpdate { document, sources })
    }
    pub(crate) fn commit_prepared(&mut self, prepared: PreparedUpdate) {
        self.document = prepared.document;
        self.sources = prepared.sources;
    }
    /// Commit document and resource descriptors together; pre-rename failure keeps old source.
    pub(crate) fn commit_atomic(&mut self, prepared: PreparedUpdate) -> Result<(), StorageError> {
        let previous_document = std::mem::replace(&mut self.document, prepared.document);
        let previous_sources = std::mem::replace(&mut self.sources, prepared.sources);
        if let Err(error) = self.autosave() {
            if error.code == "storage_durability_uncertain" {
                return Ok(());
            }
            self.document = previous_document;
            self.sources = previous_sources;
            return Err(error);
        }
        Ok(())
    }
    pub(crate) fn prune_capture_files(&self, released: &[BlobKey]) -> Result<(), StorageError> {
        if released.is_empty() {
            return Ok(());
        }
        if let Some(store) = &self.store {
            store.prune_released(released)?;
        }
        Ok(())
    }
    pub(crate) fn autosave(&mut self) -> Result<(), StorageError> {
        if self.store.is_some() {
            self.save()
        } else {
            Ok(())
        }
    }
    pub fn edit(&mut self, request: EditRequest) -> Result<(), StorageError> {
        self.ensure_writable()?;
        self.document = self.document.edited(request)?;
        if self.store.is_some() {
            self.save()
        } else {
            Ok(())
        }
    }
    /// Publish a document prepared from the current one, such as a rerun's prompt edits.
    pub(crate) fn publish(&mut self, candidate: ProjectDocument) -> Result<(), StorageError> {
        self.ensure_writable()?;
        let previous = std::mem::replace(&mut self.document, candidate);
        if let Err(error) = self.autosave() {
            // After a completed rename, keep memory aligned with the committed file.
            if error.code == "storage_durability_uncertain" {
                return Ok(());
            }
            self.document = previous;
            return Err(error);
        }
        Ok(())
    }
    /// The document `apply_authoring` would publish, validated; nothing changes.
    pub fn authoring_candidate(
        &self,
        request: EditRequest,
        mut draft: imagine_video_contract::authoring::Draft,
    ) -> Result<ProjectDocument, StorageError> {
        let mut candidate = self.document.clone();
        editing::apply(&mut candidate.project, request)?;
        super::review::invalidate_changed(&self.document.project, &mut candidate)?;
        if candidate.authoring.iter().any(|d| d.id == draft.id) {
            return Err(StorageError::new("authoring_record_duplicate"));
        }
        draft.phase = imagine_video_contract::authoring::Phase::Applied;
        candidate.authoring.push(draft);
        candidate.validate()?;
        Ok(candidate)
    }
    /// Publish reviewed source and its immutable provider/profile provenance in one revision.
    pub fn apply_authoring(
        &mut self,
        request: EditRequest,
        draft: imagine_video_contract::authoring::Draft,
    ) -> Result<(), StorageError> {
        self.ensure_writable()?;
        let candidate = self.authoring_candidate(request, draft)?;
        let previous = std::mem::replace(&mut self.document, candidate);
        if let Err(error) = self.autosave() {
            // Rename already committed this source revision. Keep it applied and let
            // the snapshot's NotSaved/error status expose uncertain durability;
            // rolling memory back here could overwrite the new manifest on retry.
            if error.code == "storage_durability_uncertain" {
                return Ok(());
            }
            self.document = previous;
            return Err(error);
        }
        Ok(())
    }
    pub fn review(&mut self, request: super::review::ReviewRequest) -> Result<(), StorageError> {
        self.ensure_writable()?;
        let candidate = super::review::apply(&self.document, request)?;
        let previous = std::mem::replace(&mut self.document, candidate);
        if let Err(error) = self.autosave() {
            // After a completed rename, keep memory aligned with the committed file.
            if error.code == "storage_durability_uncertain" {
                return Ok(());
            }
            self.document = previous;
            return Err(error);
        }
        Ok(())
    }
    /// Explicit first Save. The directory must exist and be empty.
    pub fn save_to(&mut self, path: &Path) -> Result<(), StorageError> {
        if self.store.is_some() {
            return Err(StorageError::new("project_already_bound"));
        }
        self.status.state = SaveState::Saving;
        let result = Store::select(path, "project.json");
        let mut store = match result {
            Ok(store) => store,
            Err(error) => {
                self.failed(&error);
                return Err(error);
            }
        };
        match store.save(&self.document, &mut self.sources) {
            Ok(()) => {
                self.store = Some(store);
                self.saved();
                Ok(())
            }
            Err(error) => {
                if error.code == "storage_durability_uncertain" {
                    self.store = Some(store);
                }
                self.failed(&error);
                Err(error)
            }
        }
    }

    /// Bind a newly authored session to the verified private project layout.
    /// Bound legacy projects are never relocated or converted by this method.
    pub fn save_private(
        &mut self,
        root: &crate::private_storage::Root,
    ) -> Result<(), StorageError> {
        let layout = root
            .project(&self.document.project.id.0)
            .map_err(StorageError::new);
        self.bind_private(root, layout)
    }
    /// Bind a new session to an explicitly located folder in an approved
    /// private project root. The folder must be empty; nothing is moved.
    pub fn save_private_at(
        &mut self,
        root: &crate::private_storage::Root,
        path: &Path,
    ) -> Result<(), StorageError> {
        let layout = root
            .project_location(path)
            .map_err(StorageError::new)
            .and_then(|()| {
                root.allows(path)
                    .then_some(())
                    .ok_or_else(|| StorageError::new("invalid_storage_root"))
            })
            .and_then(|()| empty_project_folder(path))
            .and_then(|()| root.project_at(path).map_err(StorageError::new));
        self.bind_private(root, layout)
    }
    fn bind_private(
        &mut self,
        root: &crate::private_storage::Root,
        layout: Result<crate::storage_layout::Project, StorageError>,
    ) -> Result<(), StorageError> {
        if self.store.is_some() {
            return Err(StorageError::new("project_already_bound"));
        }
        self.status.state = SaveState::Saving;
        let mut candidate = self.document.clone();
        candidate.storage_layout = Some(1);
        let result =
            layout.and_then(|layout| Store::select_private(&layout.root, "project.json", root));
        let mut store = match result {
            Ok(store) => store,
            Err(error) => {
                self.failed(&error);
                return Err(error);
            }
        };
        match store.save(&candidate, &mut self.sources) {
            Ok(()) => {
                self.document = candidate;
                self.store = Some(store);
                self.saved();
                Ok(())
            }
            Err(error) => {
                if error.code == "storage_durability_uncertain" {
                    self.document = candidate;
                    self.store = Some(store);
                }
                self.failed(&error);
                Err(error)
            }
        }
    }

    pub fn save(&mut self) -> Result<(), StorageError> {
        self.status.state = SaveState::Saving;
        let result = self
            .store
            .as_mut()
            .ok_or_else(|| StorageError::new("project_not_bound"))
            .and_then(|store| store.save(&self.document, &mut self.sources));
        match result {
            Ok(()) => {
                self.saved();
                Ok(())
            }
            Err(error) => {
                self.failed(&error);
                Err(error)
            }
        }
    }
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let (store, document) = Store::<ProjectDocument>::open(path, "project.json")?;
        Self::from_store(store, document)
    }
    fn from_store(
        store: Store<ProjectDocument>,
        document: ProjectDocument,
    ) -> Result<Self, StorageError> {
        let sources = store.files(&document)?;
        let revision = document.project.project_revision;
        let location = Some(store.path().to_path_buf());
        Ok(Self {
            document,
            sources,
            store: Some(store),
            status: PersistenceStatus {
                state: SaveState::Saved,
                saved_revision: Some(revision),
                error_code: None,
                location,
            },
        })
    }
    /// Reopen a private-layout project while retaining the mount proof for every save.
    pub fn open_private(
        root: &crate::private_storage::Root,
        id: &ProjectId,
    ) -> Result<Self, StorageError> {
        root.proof().map_err(StorageError::new)?;
        let layout = root.layout().project(&id.0).map_err(StorageError::new)?;
        let (store, document) =
            Store::<ProjectDocument>::open_private(&layout.root, "project.json", root)?;
        let mut session = Self::from_store(store, document)?;
        if &session.document.project.id != id || session.document.storage_layout != Some(1) {
            return Err(StorageError::new("private_project_layout_mismatch"));
        }
        session
            .store
            .as_mut()
            .expect("open binds store")
            .guard_private(root)?;
        Ok(session)
    }
    /// The saved document of a private project folder, read without opening a
    /// session; used to list projects.
    pub fn peek_private(
        root: &crate::private_storage::Root,
        path: &Path,
    ) -> Result<ProjectDocument, StorageError> {
        Store::<ProjectDocument>::open_readonly(path, "project.json", root)
            .map(|(_, document)| document)
    }
    /// Explicitly selected older projects remain readable. Outside the private
    /// root they require Save As before edits; no file is moved or migrated.
    pub fn open_for_private(
        root: &crate::private_storage::Root,
        path: &Path,
    ) -> Result<Self, StorageError> {
        root.proof().map_err(StorageError::new)?;
        let (store, document) =
            Store::<ProjectDocument>::open_readonly(path, "project.json", root)?;
        let mut session = Self::from_store(store, document)?;
        if session.document.storage_layout == Some(1) {
            let expected = root
                .layout()
                .project(&session.document.project.id.0)
                .map_err(StorageError::new)?;
            if path != expected.root && root.project_location(path).is_err() {
                return Err(StorageError::new("private_storage_location_required"));
            }
            session
                .store
                .as_mut()
                .expect("open binds store")
                .guard_private(root)?;
        } else if path.starts_with(root.layout().root()) {
            session
                .store
                .as_mut()
                .expect("open binds store")
                .guard_private(root)?;
        } else {
            session
                .store
                .as_mut()
                .expect("open binds store")
                .block_writes("legacy_project_requires_private_copy");
            session.status.error_code = Some("legacy_project_requires_private_copy".into());
        }
        Ok(session)
    }
    /// Save as switches the active location/identity only after the new bundle commits.
    pub fn save_as(&mut self, path: &Path) -> Result<(), StorageError> {
        let mut candidate = self.document.clone();
        candidate.project.id = ProjectId::generate();
        let result: Result<Store<ProjectDocument>, StorageError> = (|| {
            let mut store = Store::select(path, "project.json")?;
            store.save(&candidate, &mut self.sources)?;
            Ok(store)
        })();
        match result {
            Ok(store) => {
                self.document = candidate;
                self.store = Some(store);
                self.saved();
                Ok(())
            }
            Err(error) => {
                // A failed copy does not make the still-active original unsaved.
                self.status.error_code = Some(error.code.clone());
                Err(error)
            }
        }
    }
    /// Export a separately identified older-format copy without rebinding the session.
    pub fn export_compatible(
        &mut self,
        path: &Path,
    ) -> Result<super::compatibility::Report, StorageError> {
        let (copy, report) = super::compatibility::project(&self.document)?;
        let mut store = Store::select(path, "project.json")?;
        store.save(&copy, &mut self.sources)?;
        Ok(report)
    }
    /// Explicit older-format copy under a new fixed private identity. This does
    /// not rebind, rewrite or migrate the active project or external references.
    pub fn export_compatible_private(
        &mut self,
        root: &crate::private_storage::Root,
    ) -> Result<super::compatibility::Report, StorageError> {
        let (copy, report) = super::compatibility::project(&self.document)?;
        let layout = root
            .project(&copy.project.id.0)
            .map_err(StorageError::new)?;
        let mut store = Store::select_private(&layout.root, "project.json", root)?;
        store.save(&copy, &mut self.sources)?;
        Ok(report)
    }
    fn saved(&mut self) {
        self.status = PersistenceStatus {
            state: SaveState::Saved,
            saved_revision: Some(self.document.project.project_revision),
            error_code: None,
            location: self.location(),
        };
    }
    fn failed(&mut self, error: &StorageError) {
        self.status.state = SaveState::NotSaved;
        self.status.error_code = Some(error.code.clone());
        self.status.location = self.location();
    }
    /// The bound project folder, where every save and autosave goes.
    pub fn location(&self) -> Option<PathBuf> {
        self.store.as_ref().map(|store| store.path().to_path_buf())
    }
    pub(crate) fn files(&self) -> Result<BTreeMap<BlobKey, File>, StorageError> {
        bundle::clone_files(&self.sources)
    }
    pub(crate) fn from_copy(
        mut document: ProjectDocument,
        inputs: BTreeMap<BlobKey, File>,
        staging: &SessionStaging,
    ) -> Result<Self, StorageError> {
        document.project.id = ProjectId::generate();
        document.project.project_revision = 0;
        document.storage_layout = None;
        document.validate()?;
        let mut sources = BTreeMap::new();
        let owned: BTreeSet<_> = document
            .resources
            .iter()
            .filter(|r| r.source.is_none())
            .map(Resource::key)
            .collect();
        for (key, mut source) in inputs {
            if owned.contains(&key) {
                sources.insert(key, staging.copy(&mut source)?);
            }
        }
        for resource in &document.resources {
            if resource.source.is_some() {
                continue;
            }
            let file = sources
                .get_mut(&resource.key())
                .ok_or_else(|| StorageError::new("resource_missing"))?;
            if bundle::digest(file)? != (resource.sha256.clone(), resource.bytes) {
                return Err(StorageError::new("resource_integrity_failed"));
            }
        }
        Ok(Self {
            document,
            sources,
            store: None,
            status: PersistenceStatus {
                state: SaveState::SessionOnly,
                saved_revision: None,
                error_code: None,
                location: None,
            },
        })
    }
}

#[cfg(test)]
#[path = "private_project_tests.rs"]
mod private_project_tests;

#[cfg(test)]
use crate::task_fixtures;
#[cfg(test)]
#[path = "external_resource_tests.rs"]
mod external_resource_tests;
#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write};
    #[test]
    fn imported_take_save_failure_is_atomic_and_reopen_keeps_unknown_recipe() {
        use crate::video::captures::Import;
        let fixtures = super::task_fixtures::env_root("TMPDIR");
        let root = tempfile::tempdir_in(&fixtures).unwrap();
        let stage = tempfile::tempdir_in(&fixtures).unwrap();
        let staging = SessionStaging::select(stage.path()).unwrap();
        let original = stage.path().join("own-original.bin");
        fs::write(&original, b"OWN EXTERNAL VIDEO FIXTURE").unwrap();
        let mut session = ProjectSession::new(project()).unwrap();
        let mut document = session.document.clone();
        let edit: EditRequest = serde_json::from_str(include_str!(
            "../../design/video-contract/edit-project.json"
        ))
        .unwrap();
        editing::apply(&mut document.project, edit).unwrap();
        let scene = document.project.scenes[0].id.clone();
        let asset_id = AssetId("own-import".into());
        document.project.assets.insert(
            asset_id.clone(),
            Asset {
                id: asset_id.clone(),
                name: "Own import".into(),
                revision: 1,
                kind: MediaKind::Video,
                library_source: None,
            },
        );
        let mut input = File::open(&original).unwrap();
        let reference = Resource::inspect(
            "own-reference".into(),
            ResourceKind::Reference {
                asset_id: asset_id.clone(),
                revision: 1,
            },
            &mut input,
        )
        .unwrap();
        document.resources.push(reference.clone());
        session
            .update(
                0,
                document,
                BTreeMap::from([(reference.key(), input)]),
                &staging,
            )
            .unwrap();
        session.save_to(root.path()).unwrap();
        let request = |session: &ProjectSession| Import {
            schema_version: 1,
            expected_project_revision: session.document.project.project_revision,
            scene_id: scene.clone(),
            asset_id: asset_id.clone(),
            accept: true,
        };
        let before = fs::read(root.path().join("project.json")).unwrap();
        session.store.as_mut().unwrap().fail_commit = true;
        assert_eq!(
            session
                .import_take(request(&session), &staging)
                .err()
                .unwrap()
                .code,
            "injected_disk_full"
        );
        assert!(session.document.takes.is_empty());
        assert_eq!(fs::read(root.path().join("project.json")).unwrap(), before);
        session.store.as_mut().unwrap().fail_commit = false;
        let id = session
            .import_take(request(&session), &staging)
            .unwrap()
            .attached_id
            .unwrap();
        assert_eq!(
            session
                .import_take(request(&session), &staging)
                .unwrap()
                .attached_id,
            Some(id.clone())
        );
        let review = |session: &ProjectSession, action| super::super::review::ReviewRequest {
            schema_version: 1,
            expected_project_revision: session.document.project.project_revision,
            scene_id: scene.clone(),
            take_id: id.clone(),
            action,
        };
        let before_review = session.document.clone();
        let before_disk = fs::read(root.path().join("project.json")).unwrap();
        session.store.as_mut().unwrap().fail_commit = true;
        assert_eq!(
            session
                .review(review(&session, super::super::review::ReviewAction::Reject))
                .unwrap_err()
                .code,
            "injected_disk_full"
        );
        assert_eq!(session.document, before_review);
        assert_eq!(
            fs::read(root.path().join("project.json")).unwrap(),
            before_disk
        );
        session.store.as_mut().unwrap().fail_commit = false;
        session
            .review(review(&session, super::super::review::ReviewAction::Reject))
            .unwrap();
        assert!(session.document.project.scenes[0].accepted_take.is_none());
        session
            .review(review(&session, super::super::review::ReviewAction::Accept))
            .unwrap();
        drop(session);
        let reopened = ProjectSession::open(root.path()).unwrap();
        let take = reopened.document.takes.iter().find(|t| t.id == id).unwrap();
        assert!(take.provenance.is_none());
        let accepted = reopened.document.project.scenes[0]
            .accepted_take
            .as_ref()
            .unwrap();
        assert!(accepted.model.is_none());
        assert_eq!(accepted.source_fingerprint, reference.sha256);
        assert_eq!(
            fs::read(root.path().join("takes").join(&reference.sha256)).unwrap(),
            b"OWN EXTERNAL VIDEO FIXTURE"
        );
        assert_eq!(fs::read(original).unwrap(), b"OWN EXTERNAL VIDEO FIXTURE");
    }
    fn project() -> VideoProject {
        VideoProject::create(
            serde_json::from_str(include_str!(
                "../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn capture_delete_is_atomic_and_only_prunes_unshared_owned_blobs() {
        use crate::video::captures::{Action, Request};
        let fixtures = super::task_fixtures::env_root("TMPDIR");
        let root = tempfile::tempdir_in(&fixtures).unwrap();
        let stage = tempfile::tempdir_in(&fixtures).unwrap();
        let staging = SessionStaging::select(stage.path()).unwrap();
        let original = stage.path().join("own-external-original.bin");
        fs::write(&original, b"OWN SYNTHETIC CAPTURE").unwrap();
        let mut session = ProjectSession::new(project()).unwrap();
        let mut document = session.document.clone();
        let edit: EditRequest = serde_json::from_str(include_str!(
            "../../design/video-contract/edit-project.json"
        ))
        .unwrap();
        editing::apply(&mut document.project, edit).unwrap();
        let scene = document.project.scenes[0].id.clone();
        let model = document.project.default_model.clone();
        let mut source = File::open(&original).unwrap();
        let first = Resource::inspect(
            "owned-one".into(),
            ResourceKind::TakeVideo {
                take_id: TakeId("one".into()),
            },
            &mut source,
        )
        .unwrap();
        let key = first.key();
        for id in ["one", "two"] {
            document.takes.push(StoredTake {
                id: TakeId(id.into()),
                scene_id: scene.clone(),
                status: TakeStatus::Candidate,
                provenance: Some(TakeProvenance {
                    boundary: None,
                    resolved_loras: None,
                    model: model.clone(),
                    seed: 42,
                    input_revision: "own-input".into(),
                    compiled_prompt: "Own synthetic receipt".into(),
                    settings: SettingsSlot {
                        model: model.clone(),
                        schema_version: 1,
                        values: serde_json::json!({}),
                    },
                    references: vec![],
                    source_take: None,
                    continuity: None,
                    authored: None,
                }),
            });
            document.resources.push(Resource {
                id: format!("owned-{id}"),
                kind: ResourceKind::TakeVideo {
                    take_id: TakeId(id.into()),
                },
                ..first.clone()
            });
        }
        session
            .update(
                0,
                document,
                BTreeMap::from([(key.clone(), source)]),
                &staging,
            )
            .unwrap();
        session.save_to(root.path()).unwrap();
        let owned = root.path().join("takes").join(&key.sha256);
        let request = |session: &ProjectSession, id: &str| Request {
            schema_version: 1,
            expected_project_revision: session.document.project.project_revision,
            scene_id: scene.clone(),
            take_id: TakeId(id.into()),
            action: Action::Delete,
        };
        let before = fs::read(root.path().join("project.json")).unwrap();
        session.store.as_mut().unwrap().fail_commit = true;
        assert_eq!(
            session
                .capture(request(&session, "one"), &staging)
                .err()
                .unwrap()
                .code,
            "injected_disk_full"
        );
        assert_eq!(session.document.takes.len(), 2);
        assert_eq!(fs::read(root.path().join("project.json")).unwrap(), before);
        assert!(owned.exists());
        session.store.as_mut().unwrap().fail_commit = false;
        assert!(
            session
                .capture(request(&session, "one"), &staging)
                .unwrap()
                .cleanup_warning
                .is_none()
        );
        assert!(
            owned.exists(),
            "A second Take still owns this content-addressed blob"
        );
        assert!(
            session
                .capture(request(&session, "two"), &staging)
                .unwrap()
                .cleanup_warning
                .is_none()
        );
        assert!(!owned.exists());
        assert_eq!(fs::read(&original).unwrap(), b"OWN SYNTHETIC CAPTURE");
        assert!(
            ProjectSession::open(root.path())
                .unwrap()
                .document
                .takes
                .is_empty()
        );
    }
    #[test]
    fn media_relink_precommit_failure_rolls_back_and_postrename_keeps_both_versions() {
        use super::super::media_health::{Relink, apply_relink};
        let fixture_root = super::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES");
        let stage = tempfile::tempdir_in(&fixture_root).unwrap();
        let staging = SessionStaging::select(stage.path()).unwrap();
        let saved = tempfile::tempdir_in(&fixture_root).unwrap();
        let mut session = ProjectSession::new(project()).unwrap();
        let id = AssetId("own-relink".into());
        let mut document = session.document.clone();
        document.project.assets.insert(
            id.clone(),
            Asset {
                library_source: None,
                name: "Own".into(),
                id: id.clone(),
                revision: 1,
                kind: MediaKind::Image,
            },
        );
        let mut old = tempfile::tempfile_in(&fixture_root).unwrap();
        old.write_all(b"OWN OLD RESOURCE").unwrap();
        let resource = Resource::inspect(
            "own-old".into(),
            ResourceKind::Reference {
                asset_id: id.clone(),
                revision: 1,
            },
            &mut old,
        )
        .unwrap();
        document.resources.push(resource.clone());
        session
            .update(
                0,
                document,
                BTreeMap::from([(resource.key(), old)]),
                &staging,
            )
            .unwrap();
        session.save_to(saved.path()).unwrap();
        let original = session.document.clone();
        let request = Relink {
            schema_version: 1,
            expected_project_revision: 1,
            asset_id: id.clone(),
            expected_asset_revision: 1,
            replacement_asset_id: "own-upload".into(),
        };
        let new = || {
            let mut file = tempfile::tempfile_in(&fixture_root).unwrap();
            file.write_all(b"OWN NEW RESOURCE").unwrap();
            file
        };
        session.store.as_mut().unwrap().fail_commit = true;
        assert_eq!(
            apply_relink(&mut session, &request, new(), &MediaKind::Image, &staging)
                .unwrap_err()
                .code,
            "injected_disk_full"
        );
        assert_eq!(session.document, original);
        assert_eq!(
            ProjectSession::open(saved.path()).unwrap().document,
            original
        );
        let mut old = vec![];
        session.copy_resource("own-old", &mut old).unwrap();
        assert_eq!(old, b"OWN OLD RESOURCE");
        session.store.as_mut().unwrap().fail_commit = false;
        session
            .store
            .as_mut()
            .unwrap()
            .inject_sync_failure_after_rename(true);
        apply_relink(&mut session, &request, new(), &MediaKind::Image, &staging).unwrap();
        assert_eq!(session.document.project.assets[&id].revision, 2);
        assert_eq!(session.document.resources.len(), 2);
        assert_eq!(session.status.state, SaveState::NotSaved);
        assert_eq!(
            session.status.error_code.as_deref(),
            Some("storage_durability_uncertain")
        );
        assert_eq!(
            ProjectSession::open(saved.path()).unwrap().document,
            session.document
        );
    }
    #[test]
    fn authoring_precommit_failure_rolls_back_but_postrename_keeps_applied_source() {
        use imagine_video_contract::authoring::*;
        use imagine_video_contract::editing::Edit;
        let root = tempfile::tempdir().unwrap();
        let mut session = ProjectSession::new(project()).unwrap();
        session.save_to(root.path()).unwrap();
        let original = session.document.clone();
        let request = || EditRequest {
            schema_version: 1,
            expected_project_revision: 0,
            commands: vec![Edit::SetGlobalPrompt {
                prompt: "Own reviewed source".into(),
            }],
        };
        let draft = Draft {
            schema_version: 1,
            id: "own_applied_draft".into(),
            project_id: original.project.id.clone(),
            request: Request {
                schema_version: 1,
                expected_project_revision: 0,
                provider_profile_id: "own_fixture".into(),
                provider: "synthetic".into(),
                prompt_profile_id: "normal".into(),
                operation: Operation::Outline,
                scene_field: None,
                scene_id: None,
                section_id: None,
                brief: String::new(),
                instruction: String::new(),
                duration_seconds: 5.,
                destination: Destination::Append,
                cast: vec![],
                images: vec![],
                add_dialogue: false,
                add_music: false,
            },
            phase: Phase::Review,
            outline: None,
            scenes: vec![],
            text: Some("Own reviewed source".into()),
            failure: None,
            failed_index: None,
            prompt_profile_id: "normal".into(),
            prompt_profile_revision: "own-test".into(),
            profile_warning: String::new(),
            provider_snapshot: serde_json::json!({"provider":"synthetic"}),
            planned_durations: vec![],
            inventory: vec![],
            selected_cast: serde_json::Value::Null,
            warnings: vec![],
            fields: vec![],
            production: None,
        };
        session.store.as_mut().unwrap().fail_commit = true;
        assert_eq!(
            session
                .apply_authoring(request(), draft.clone())
                .unwrap_err()
                .code,
            "injected_disk_full"
        );
        assert_eq!(session.document.project, original.project);
        assert!(session.document.authoring.is_empty());
        assert_eq!(
            ProjectSession::open(root.path()).unwrap().document.project,
            original.project
        );
        assert_eq!(session.status.state, SaveState::NotSaved);
        session.store.as_mut().unwrap().fail_commit = false;
        session
            .store
            .as_mut()
            .unwrap()
            .inject_sync_failure_after_rename(true);
        session.apply_authoring(request(), draft).unwrap();
        assert_eq!(
            session.document.project.global_prompt,
            "Own reviewed source"
        );
        assert_eq!(session.document.project.project_revision, 1);
        assert_eq!(session.document.authoring[0].phase, Phase::Applied);
        assert_eq!(session.status.state, SaveState::NotSaved);
        assert_eq!(session.status.saved_revision, Some(0));
        assert_eq!(
            session.status.error_code.as_deref(),
            Some("storage_durability_uncertain")
        );
        let disk = ProjectSession::open(root.path()).unwrap();
        assert_eq!(disk.document.project, session.document.project);
        assert_eq!(disk.document.authoring, session.document.authoring);
        session
            .store
            .as_mut()
            .unwrap()
            .inject_sync_failure_after_rename(false);
        session.save().unwrap();
        assert_eq!(session.status.state, SaveState::Saved);
        assert_eq!(session.status.saved_revision, Some(1));
        session
            .edit(EditRequest {
                schema_version: 1,
                expected_project_revision: 1,
                commands: vec![Edit::SetGlobalPrompt {
                    prompt: "Own subsequent edit".into(),
                }],
            })
            .unwrap();
        let disk = ProjectSession::open(root.path()).unwrap();
        assert_eq!(disk.document.project.project_revision, 2);
        assert_eq!(disk.document.project.global_prompt, "Own subsequent edit");
        assert_eq!(disk.document.authoring, session.document.authoring);
        assert_eq!(disk.document.authoring.len(), 1);
    }
    #[test]
    fn interrupted_autosave_keeps_manifest_and_session_draft_for_retry() {
        let root = tempfile::tempdir().unwrap();
        let stage = tempfile::tempdir().unwrap();
        let staging = SessionStaging::select(stage.path()).unwrap();
        let mut session = ProjectSession::new(project()).unwrap();
        session.save_to(root.path()).unwrap();
        let previous = fs::read(root.path().join("project.json")).unwrap();
        let mut document = session.document.clone();
        document.project.global_prompt = "UNSAVED SYNTHETIC".into();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"NEW SYNTHETIC MEDIA").unwrap();
        let resource = Resource::inspect(
            "new".into(),
            ResourceKind::Reference {
                asset_id: AssetId("asset".into()),
                revision: 1,
            },
            &mut file,
        )
        .unwrap();
        let key = resource.key();
        document.resources.push(resource);
        session.store.as_mut().unwrap().fail_commit = true;
        assert!(
            session
                .update(0, document, BTreeMap::from([(key.clone(), file)]), &staging)
                .is_err()
        );
        assert_eq!(session.status.state, SaveState::NotSaved);
        assert_eq!(session.status.saved_revision, Some(0));
        assert_eq!(
            fs::read(root.path().join("project.json")).unwrap(),
            previous
        );
        assert!(!root.path().join("media").join(&key.sha256).exists());
        assert_eq!(
            ProjectSession::open(root.path())
                .unwrap()
                .document
                .project
                .global_prompt,
            ""
        );
        session.store.as_mut().unwrap().fail_commit = false;
        session.save().unwrap();
        assert_eq!(
            ProjectSession::open(root.path())
                .unwrap()
                .document
                .project
                .global_prompt,
            "UNSAVED SYNTHETIC"
        );
    }
    #[test]
    fn failed_first_save_can_retry_and_lock_rejects_concurrent_writer() {
        let root = tempfile::tempdir().unwrap();
        let mut store = Store::select(root.path(), "project.json").unwrap();
        let document = ProjectDocument::new(project());
        let mut sources = BTreeMap::new();
        store.fail_commit = true;
        assert!(store.save(&document, &mut sources).is_err());
        assert!(!root.path().join("project.json").exists());
        store.fail_commit = false;
        store.save(&document, &mut sources).unwrap();
        let lockroot = Directory::select(root.path()).unwrap();
        let _held = lockroot.lock().unwrap();
        assert_eq!(
            store.save(&document, &mut sources).unwrap_err().code,
            "storage_busy"
        );
    }
}
