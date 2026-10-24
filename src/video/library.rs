//! Separate, explicitly saved local library. Entries never remain live project dependencies.
use super::{
    bundle::{self, Bundle, Store},
    contract::*,
    persistence::*,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::Path,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceProfile {
    pub language: String,
    pub description: String,
    pub reference_ids: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterDocument {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub alias: String,
    pub description: String,
    pub tags: Vec<String>,
    pub visual_ids: Vec<String>,
    pub voice: Option<VoiceProfile>,
    pub resources: Vec<Resource>,
    /// Portable reference metadata: image/voice asset keys refer to resource ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<imagine_video_contract::creative::VisualReference>,
}
impl CharacterDocument {
    fn validate(&self) -> Result<(), StorageError> {
        if self.schema_version != 1 || self.id.trim().is_empty() || self.name.trim().is_empty() {
            return Err(StorageError::new("invalid_character"));
        }
        validate_resources(&self.resources)?;
        let ids: BTreeSet<_> = self.resources.iter().map(|r| &r.id).collect();
        if self
            .visual_ids
            .iter()
            .chain(self.voice.iter().flat_map(|v| v.reference_ids.iter()))
            .any(|id| !ids.contains(id))
        {
            return Err(StorageError::new("character_resource_missing"));
        }
        if self
            .resources
            .iter()
            .any(|r| !matches!(r.kind, ResourceKind::Reference { .. }))
        {
            return Err(StorageError::new("invalid_character_resource"));
        }
        if let Some(reference) = &self.reference
            && (reference
                .images
                .iter()
                .any(|image| !self.visual_ids.contains(&image.asset.0))
                || reference
                    .voice
                    .iter()
                    .flat_map(|voice| &voice.assets)
                    .any(|asset| {
                        self.voice
                            .as_ref()
                            .is_none_or(|voice| !voice.reference_ids.contains(&asset.0))
                    }))
        {
            return Err(StorageError::new("character_resource_missing"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "document",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum LibraryContent {
    Project(Box<ProjectDocument>),
    Character(Box<CharacterDocument>),
    Asset(Box<AssetDocument>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetDocument {
    pub asset: Asset,
    pub resource: Resource,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    pub id: String,
    pub revision: u64,
    pub title: String,
    pub content: LibraryContent,
    /// When the entry was last saved or edited, in milliseconds since the
    /// epoch; 0 for entries saved before it was recorded.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub updated_ms: u64,
    /// Tags and description of an asset or project; a character keeps its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
/// Trimmed, non-empty, unique tags, as the Library filters them.
fn clean_tags(tags: Vec<String>) -> Result<Vec<String>, StorageError> {
    let mut out: Vec<String> = vec![];
    for tag in tags {
        let tag = tag.trim().to_owned();
        if tag.chars().count() > 64 {
            return Err(StorageError::new("invalid_library_tag"));
        }
        if !tag.is_empty() && !out.iter().any(|t| t.eq_ignore_ascii_case(&tag)) {
            out.push(tag);
        }
    }
    if out.len() > 64 {
        return Err(StorageError::new("invalid_library_tag"));
    }
    Ok(out)
}
impl LibraryEntry {
    fn resources(&self) -> Vec<&Resource> {
        match &self.content {
            LibraryContent::Project(p) => p.resources.iter().collect(),
            LibraryContent::Character(c) => c.resources.iter().collect(),
            LibraryContent::Asset(a) => vec![&a.resource],
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryDocument {
    pub schema_version: u32,
    pub id: String,
    pub revision: u64,
    pub entries: Vec<LibraryEntry>,
}
impl Bundle for LibraryDocument {
    fn identity(&self) -> &str {
        &self.id
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn resources(&self) -> Vec<&Resource> {
        self.entries
            .iter()
            .flat_map(LibraryEntry::resources)
            .collect()
    }
    fn validate(&self) -> Result<(), StorageError> {
        if self.schema_version != 1 || self.id.trim().is_empty() || self.revision > MAX_INTEGER {
            return Err(StorageError::new("invalid_library"));
        }
        let mut ids = BTreeSet::new();
        for entry in &self.entries {
            if entry.id.trim().is_empty()
                || entry.revision == 0
                || entry.revision > MAX_INTEGER
                || !ids.insert(&entry.id)
                || entry.title.trim().is_empty()
            {
                return Err(StorageError::new("invalid_library_entry"));
            }
            match &entry.content {
                LibraryContent::Project(p) => p.validate()?,
                LibraryContent::Character(c) => c.validate()?,
                LibraryContent::Asset(a) => {
                    validate_resources(std::slice::from_ref(&a.resource))?;
                    if !matches!(&a.resource.kind,ResourceKind::Reference{asset_id,..}if asset_id==&a.asset.id)
                    {
                        return Err(StorageError::new("asset_resource_missing"));
                    }
                }
            }
        }
        Ok(())
    }
}
pub struct CharacterCopy {
    pub character: CharacterDocument,
    pub origin: LibraryOrigin,
    pub sources: BTreeMap<BlobKey, File>,
}
#[derive(Debug, Clone, Serialize)]
pub struct MediaEntry {
    pub entry_id: String,
    pub entry_revision: u64,
    pub resource_id: String,
    pub name: String,
    pub kind: MediaKind,
    /// What saved it: an asset, a character or a project.
    pub entry_kind: &'static str,
    pub updated_ms: u64,
}
pub struct LibrarySession {
    document: LibraryDocument,
    store: Store<LibraryDocument>,
    sources: BTreeMap<BlobKey, File>,
    location: std::path::PathBuf,
}
impl LibrarySession {
    pub fn create(path: &Path) -> Result<Self, StorageError> {
        let store = Store::select(path, "library.json")?;
        Self::create_at(path, store)
    }
    fn create_at(path: &Path, mut store: Store<LibraryDocument>) -> Result<Self, StorageError> {
        let document = LibraryDocument {
            schema_version: 1,
            id: format!("library_{}", uuid::Uuid::new_v4().simple()),
            revision: 0,
            entries: vec![],
        };
        let mut sources = BTreeMap::new();
        store.save(&document, &mut sources)?;
        Ok(Self {
            document,
            store,
            sources,
            location: path.into(),
        })
    }
    pub fn create_private(root: &crate::private_storage::Root) -> Result<Self, StorageError> {
        let path = root.library().map_err(StorageError::new)?;
        let store = Store::select_private(&path, "library.json", root)?;
        Self::create_at(&path, store)
    }
    pub fn open_private(root: &crate::private_storage::Root) -> Result<Self, StorageError> {
        let path = root.library().map_err(StorageError::new)?;
        let (store, document) =
            Store::<LibraryDocument>::open_private(&path, "library.json", root)?;
        let sources = store.files(&document)?;
        Ok(Self {
            document,
            store,
            sources,
            location: path,
        })
    }
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let (store, document) = Store::<LibraryDocument>::open(path, "library.json")?;
        let sources = store.files(&document)?;
        Ok(Self {
            document,
            store,
            sources,
            location: path.into(),
        })
    }
    pub fn location(&self) -> &Path {
        &self.location
    }
    pub fn document(&self) -> &LibraryDocument {
        &self.document
    }
    pub fn media_entries(&self) -> Vec<MediaEntry> {
        self.document
            .entries
            .iter()
            .flat_map(|entry| {
                entry.resources().into_iter().filter_map(move |resource| {
                    let ResourceKind::Reference { asset_id, .. } = &resource.kind else {
                        return None;
                    };
                    let (kind, name) = match &entry.content {
                        LibraryContent::Project(p) => {
                            let asset = p.project.assets.get(asset_id)?;
                            (
                                asset.kind.clone(),
                                if asset.name.is_empty() {
                                    format!("{} · {}", entry.title, asset.id.0)
                                } else {
                                    asset.name.clone()
                                },
                            )
                        }
                        LibraryContent::Character(c) => {
                            let kind = if c.visual_ids.contains(&resource.id) {
                                MediaKind::Image
                            } else if c
                                .voice
                                .as_ref()
                                .is_some_and(|v| v.reference_ids.contains(&resource.id))
                            {
                                MediaKind::Audio
                            } else {
                                return None;
                            };
                            (kind, format!("{} · {}", entry.title, resource.id))
                        }
                        LibraryContent::Asset(a) => (a.asset.kind.clone(), entry.title.clone()),
                    };
                    Some(MediaEntry {
                        entry_id: entry.id.clone(),
                        entry_revision: entry.revision,
                        resource_id: resource.id.clone(),
                        name,
                        kind,
                        entry_kind: match &entry.content {
                            LibraryContent::Project(_) => "project",
                            LibraryContent::Character(_) => "character",
                            LibraryContent::Asset(_) => "asset",
                        },
                        updated_ms: entry.updated_ms,
                    })
                })
            })
            .collect()
    }
    pub fn media_file(
        &self,
        entry: &str,
        revision: u64,
        resource: &str,
    ) -> Result<(Resource, File), StorageError> {
        use std::os::fd::AsRawFd;
        let entry = self.entry(entry, revision)?;
        let resource = entry
            .resources()
            .into_iter()
            .find(|r| r.id == resource && matches!(r.kind, ResourceKind::Reference { .. }))
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        if let Some(file) = resource.open_source()? {
            return Ok((resource.clone(), file));
        }
        let file = self
            .sources
            .get(&resource.key())
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        // Reopen this owned descriptor for an independent stream position.
        let file = File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .map_err(|_| StorageError::new("storage_io_failed"))?;
        Ok((resource.clone(), file))
    }
    /// Explicit save only. replace=(entry id, expected revision) updates a selected entry.
    pub fn save_asset(
        &mut self,
        title: String,
        replace: Option<(&str, u64)>,
        project: &ProjectSession,
        id: &AssetId,
    ) -> Result<String, StorageError> {
        let asset = project
            .document()
            .project
            .assets
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::new("asset_not_found"))?;
        let resource = project
            .document()
            .resources
            .iter()
            .find(|r| matches!(&r.kind,ResourceKind::Reference{asset_id,..}if asset_id==id))
            .cloned()
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        self.save_content(
            title,
            replace,
            LibraryContent::Asset(Box::new(AssetDocument { asset, resource })),
            project.files()?,
        )
    }
    /// Manifest-only editing/removal. External originals and content blobs are never deleted.
    pub fn edit_entry(
        &mut self,
        id: &str,
        expected: u64,
        title: Option<String>,
        description: Option<String>,
        tags: Option<Vec<String>>,
        remove: bool,
    ) -> Result<(), StorageError> {
        let tags = tags.map(clean_tags).transpose()?;
        self.entry(id, expected)?;
        let mut candidate = self.document.clone();
        candidate.revision = candidate
            .revision
            .checked_add(1)
            .ok_or_else(|| StorageError::new("revision_exhausted"))?;
        if remove {
            candidate.entries.retain(|entry| entry.id != id);
        } else {
            let entry = candidate
                .entries
                .iter_mut()
                .find(|entry| entry.id == id)
                .unwrap();
            entry.revision = entry
                .revision
                .checked_add(1)
                .ok_or_else(|| StorageError::new("revision_exhausted"))?;
            if let Some(title) = title {
                entry.title = title;
            }
            entry.updated_ms = now_ms();
            match &mut entry.content {
                LibraryContent::Character(c) => {
                    if let Some(description) = description {
                        c.description = description;
                    }
                    if let Some(tags) = tags {
                        c.tags = tags;
                    }
                }
                _ => {
                    if let Some(description) = description {
                        entry.description = description;
                    }
                    if let Some(tags) = tags {
                        entry.tags = tags;
                    }
                }
            }
        }
        if !remove {
            let index = candidate
                .entries
                .iter()
                .position(|entry| entry.id == id)
                .unwrap();
            let entry = candidate.entries.remove(index);
            candidate.entries.push(entry);
        }
        candidate.validate()?;
        let mut sources = bundle::clone_files(&self.sources)?;
        let result = self.store.save(&candidate, &mut sources);
        if result.is_ok()
            || result
                .as_ref()
                .is_err_and(|e| e.code == "storage_durability_uncertain")
        {
            self.document = candidate;
            self.sources = sources;
        }
        result?;
        self.sources = self.store.files(&self.document)?;
        Ok(())
    }
    /// A copy of an entry under a new id, sharing its stored media.
    pub fn duplicate_entry(&mut self, id: &str, expected: u64) -> Result<String, StorageError> {
        let source = self.entry(id, expected)?.clone();
        let mut candidate = self.document.clone();
        candidate.revision = candidate
            .revision
            .checked_add(1)
            .filter(|r| *r <= MAX_INTEGER)
            .ok_or_else(|| StorageError::new("revision_exhausted"))?;
        let copy = format!("entry_{}", uuid::Uuid::new_v4().simple());
        candidate.entries.push(LibraryEntry {
            id: copy.clone(),
            revision: 1,
            title: format!("{} copy", source.title),
            updated_ms: now_ms(),
            ..source
        });
        candidate.validate()?;
        let mut sources = bundle::clone_files(&self.sources)?;
        let result = self.store.save(&candidate, &mut sources);
        if result.is_ok()
            || result
                .as_ref()
                .is_err_and(|e| e.code == "storage_durability_uncertain")
        {
            self.document = candidate;
            self.sources = sources;
        }
        result?;
        self.sources = self.store.files(&self.document)?;
        Ok(copy)
    }
    pub fn save_project(
        &mut self,
        title: String,
        replace: Option<(&str, u64)>,
        project: &ProjectSession,
    ) -> Result<String, StorageError> {
        self.save_content(
            title,
            replace,
            LibraryContent::Project(Box::new(project.document().clone())),
            project.files()?,
        )
    }
    pub fn save_character(
        &mut self,
        title: String,
        replace: Option<(&str, u64)>,
        character: CharacterDocument,
        inputs: BTreeMap<BlobKey, File>,
    ) -> Result<String, StorageError> {
        character.validate()?;
        self.save_content(
            title,
            replace,
            LibraryContent::Character(Box::new(character)),
            inputs,
        )
    }
    fn save_content(
        &mut self,
        title: String,
        replace: Option<(&str, u64)>,
        content: LibraryContent,
        inputs: BTreeMap<BlobKey, File>,
    ) -> Result<String, StorageError> {
        let mut candidate = self.document.clone();
        candidate.revision = candidate
            .revision
            .checked_add(1)
            .filter(|r| *r <= MAX_INTEGER)
            .ok_or_else(|| StorageError::new("revision_exhausted"))?;
        let (id, revision) = if let Some((id, expected)) = replace {
            let existing = candidate
                .entries
                .iter()
                .find(|entry| entry.id == id)
                .ok_or_else(|| StorageError::new("entry_not_found"))?;
            if existing.revision != expected {
                return Err(StorageError::new("revision_conflict"));
            }
            if std::mem::discriminant(&existing.content) != std::mem::discriminant(&content) {
                return Err(StorageError::new("entry_kind_mismatch"));
            }
            (
                id.to_owned(),
                expected
                    .checked_add(1)
                    .filter(|r| *r <= MAX_INTEGER)
                    .ok_or_else(|| StorageError::new("revision_exhausted"))?,
            )
        } else {
            (format!("entry_{}", uuid::Uuid::new_v4().simple()), 1)
        };
        // A replaced entry keeps the tags and description given to it here.
        let (tags, description) = candidate
            .entries
            .iter()
            .find(|entry| entry.id == id)
            .map(|e| (e.tags.clone(), e.description.clone()))
            .unwrap_or_default();
        candidate.entries.retain(|entry| entry.id != id);
        candidate.entries.push(LibraryEntry {
            id: id.clone(),
            revision,
            title,
            content,
            updated_ms: now_ms(),
            tags,
            description,
        });
        candidate.validate()?;
        let mut sources = bundle::clone_files(&self.sources)?;
        sources.extend(inputs);
        let result = self.store.save(&candidate, &mut sources);
        if result.is_ok()
            || result
                .as_ref()
                .is_err_and(|e| e.code == "storage_durability_uncertain")
        {
            self.document = candidate;
            self.sources = sources;
        }
        result?;
        self.sources = self.store.files(&self.document)?;
        Ok(id)
    }
    fn entry(&self, id: &str, expected: u64) -> Result<&LibraryEntry, StorageError> {
        let entry = self
            .document
            .entries
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| StorageError::new("entry_not_found"))?;
        if entry.revision != expected {
            return Err(StorageError::new("revision_conflict"));
        }
        Ok(entry)
    }
    fn origin(&self, entry: &LibraryEntry) -> LibraryOrigin {
        LibraryOrigin {
            library_id: self.document.id.clone(),
            entry_id: entry.id.clone(),
            entry_revision: entry.revision,
        }
    }
    pub fn copy_project(
        &self,
        id: &str,
        expected: u64,
        staging: &SessionStaging,
    ) -> Result<ProjectSession, StorageError> {
        let entry = self.entry(id, expected)?;
        let LibraryContent::Project(project) = &entry.content else {
            return Err(StorageError::new("entry_kind_mismatch"));
        };
        let mut document = (**project).clone();
        document.origin = Some(self.origin(entry));
        ProjectSession::from_copy(document, self.select_sources(entry)?, staging)
    }
    pub fn copy_character(
        &self,
        id: &str,
        expected: u64,
        staging: &SessionStaging,
    ) -> Result<CharacterCopy, StorageError> {
        let entry = self.entry(id, expected)?;
        let LibraryContent::Character(character) = &entry.content else {
            return Err(StorageError::new("entry_kind_mismatch"));
        };
        let mut sources = BTreeMap::new();
        let owned: BTreeSet<_> = character
            .resources
            .iter()
            .filter(|r| r.source.is_none())
            .map(Resource::key)
            .collect();
        for (key, mut source) in self.select_sources(entry)? {
            let file = if owned.contains(&key) {
                staging.copy(&mut source)?
            } else {
                source
            };
            sources.insert(key, file);
        }
        for resource in &character.resources {
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
        Ok(CharacterCopy {
            character: (**character).clone(),
            origin: self.origin(entry),
            sources,
        })
    }
    fn select_sources(
        &self,
        entry: &LibraryEntry,
    ) -> Result<BTreeMap<BlobKey, File>, StorageError> {
        let mut files = BTreeMap::new();
        for resource in entry.resources() {
            if let Some(file) = resource.open_source()? {
                files.insert(resource.key(), file);
                continue;
            }
            let file = self
                .sources
                .get(&resource.key())
                .ok_or_else(|| StorageError::new("resource_missing"))?;
            files.insert(
                resource.key(),
                file.try_clone()
                    .map_err(|_| StorageError::new("storage_io_failed"))?,
            );
        }
        Ok(files)
    }
}
