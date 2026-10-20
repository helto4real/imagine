//! Explicit project persistence. All filesystem access starts at caller-selected roots.
use super::{
    bundle::{self, Bundle, Directory, Store},
    contract::*,
    editing::{self, EditRequest},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::Path,
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
}
impl Resource {
    pub fn key(&self) -> BlobKey {
        BlobKey {
            bucket: match self.kind {
                ResourceKind::Reference { .. } => Bucket::Media,
                ResourceKind::TakeVideo { .. } | ResourceKind::TakeAudio { .. } => Bucket::Takes,
                ResourceKind::Continuation { .. } => Bucket::Continuity,
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
        };
        resource.validate()?;
        Ok(resource)
    }
    fn validate(&self) -> Result<(), StorageError> {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TakeStatus {
    Candidate,
    Accepted,
    Rejected,
    Outdated,
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
    pub model: ModelSelection,
    pub seed: u64,
    pub input_revision: String,
    pub compiled_prompt: String,
    pub settings: SettingsSlot,
    pub references: Vec<AssetVersion>,
    pub source_take: Option<TakeId>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredTake {
    pub id: TakeId,
    pub scene_id: SceneId,
    pub status: TakeStatus,
    pub provenance: TakeProvenance,
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
    pub project: VideoProject,
    pub resources: Vec<Resource>,
    pub takes: Vec<StoredTake>,
    pub origin: Option<LibraryOrigin>,
}
impl ProjectDocument {
    pub fn new(project: VideoProject) -> Self {
        Self {
            schema_version: 1,
            project,
            resources: vec![],
            takes: vec![],
            origin: None,
        }
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
    fn validate(&self) -> Result<(), StorageError> {
        if self.schema_version != 1 {
            return Err(StorageError::new("unsupported_storage_schema"));
        }
        self.project.validate()?;
        validate_resources(&self.resources)?;
        let reference = |asset: &AssetId, revision: u64| {
            self.resources.iter().any(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,revision:v} if asset_id==asset&&*v==revision))
        };
        for asset in self.project.assets.values() {
            if !reference(&asset.id, asset.revision) {
                return Err(StorageError::new("reference_resource_missing"));
            }
        }
        let mut take_ids = BTreeSet::new();
        for take in &self.takes {
            if take.id.0.trim().is_empty()
                || !take_ids.insert(&take.id)
                || self.project.scene(&take.scene_id).is_err()
                || take.provenance.seed > MAX_INTEGER
                || take.provenance.model != take.provenance.settings.model
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
            for asset in &take.provenance.references {
                if !reference(&asset.asset_id, asset.revision) {
                    return Err(StorageError::new("historical_reference_missing"));
                }
            }
            if let Some(source) = &take.provenance.source_take
                && !self.takes.iter().any(|t| &t.id == source)
            {
                return Err(StorageError::new("source_take_missing"));
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
                        || selected.provenance.model != summary.model
                        || selected.provenance.input_revision != summary.source_fingerprint
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
                | ResourceKind::Continuation { take_id, .. } => take_id,
            };
            if !take_ids.contains(take_id) {
                return Err(StorageError::new("resource_take_missing"));
            }
        }
        Ok(())
    }
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
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveState {
    SessionOnly,
    Saving,
    Saved,
    NotSaved,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceStatus {
    pub state: SaveState,
    pub saved_revision: Option<u64>,
    pub error_code: Option<String>,
}
/// All mutation/save methods require exclusive access. No background task is launched.
/// A caller may debounce edits; update() autosaves synchronously once Save has bound a root.
pub struct ProjectSession {
    document: ProjectDocument,
    sources: BTreeMap<BlobKey, File>,
    store: Option<Store<ProjectDocument>>,
    status: PersistenceStatus,
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
            },
        })
    }
    pub fn document(&self) -> &ProjectDocument {
        &self.document
    }
    /// Stream a verified resource to a caller-owned destination; no internal path escapes.
    pub fn copy_resource(
        &self,
        id: &str,
        target: &mut impl std::io::Write,
    ) -> Result<(), StorageError> {
        let resource = self
            .document
            .resources
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        let mut source = self
            .sources
            .get(&resource.key())
            .ok_or_else(|| StorageError::new("resource_missing"))?
            .try_clone()
            .map_err(|_| StorageError::new("storage_io_failed"))?;
        if bundle::digest(&mut source)? != (resource.sha256.clone(), resource.bytes) {
            return Err(StorageError::new("resource_integrity_failed"));
        }
        bundle::stream(&source, target)?;
        Ok(())
    }
    pub fn status(&self) -> &PersistenceStatus {
        &self.status
    }
    /// Stage independent copies before changing the document. Existing sources are retained
    /// only while referenced by the new document, including historical Take references.
    pub fn update(
        &mut self,
        expected_revision: u64,
        mut document: ProjectDocument,
        inputs: BTreeMap<BlobKey, File>,
        staging: &SessionStaging,
    ) -> Result<(), StorageError> {
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
        for (key, mut file) in inputs {
            sources.insert(key, staging.copy(&mut file)?);
        }
        for resource in &document.resources {
            let file = sources
                .get_mut(&resource.key())
                .ok_or_else(|| StorageError::new("resource_missing"))?;
            if bundle::digest(file)? != (resource.sha256.clone(), resource.bytes) {
                return Err(StorageError::new("resource_integrity_failed"));
            }
        }
        let keys: BTreeSet<_> = document.resources.iter().map(Resource::key).collect();
        sources.retain(|key, _| keys.contains(key));
        self.document = document;
        self.sources = sources;
        if self.store.is_some() {
            self.save()
        } else {
            Ok(())
        }
    }
    pub fn edit(&mut self, request: EditRequest) -> Result<(), StorageError> {
        let mut candidate = self.document.clone();
        editing::apply(&mut candidate.project, request)?;
        candidate.validate()?;
        self.document = candidate;
        if self.store.is_some() {
            self.save()
        } else {
            Ok(())
        }
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
        let sources = store.files(&document)?;
        let revision = document.project.project_revision;
        Ok(Self {
            document,
            sources,
            store: Some(store),
            status: PersistenceStatus {
                state: SaveState::Saved,
                saved_revision: Some(revision),
                error_code: None,
            },
        })
    }
    /// Save as switches the active location/identity only after the new bundle commits.
    pub fn save_as(&mut self, path: &Path) -> Result<(), StorageError> {
        let mut candidate = self.document.clone();
        candidate.project.id = ProjectId(format!("project_{}", uuid::Uuid::new_v4().simple()));
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
    fn saved(&mut self) {
        self.status = PersistenceStatus {
            state: SaveState::Saved,
            saved_revision: Some(self.document.project.project_revision),
            error_code: None,
        };
    }
    fn failed(&mut self, error: &StorageError) {
        self.status.state = SaveState::NotSaved;
        self.status.error_code = Some(error.code.clone());
    }
    pub(crate) fn files(&self) -> Result<BTreeMap<BlobKey, File>, StorageError> {
        bundle::clone_files(&self.sources)
    }
    pub(crate) fn from_copy(
        mut document: ProjectDocument,
        inputs: BTreeMap<BlobKey, File>,
        staging: &SessionStaging,
    ) -> Result<Self, StorageError> {
        document.project.id = ProjectId(format!("project_{}", uuid::Uuid::new_v4().simple()));
        document.project.project_revision = 0;
        document.validate()?;
        let mut sources = BTreeMap::new();
        for (key, mut source) in inputs {
            sources.insert(key, staging.copy(&mut source)?);
        }
        for resource in &document.resources {
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
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write};
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
