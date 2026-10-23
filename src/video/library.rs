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
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    pub id: String,
    pub revision: u64,
    pub title: String,
    pub content: LibraryContent,
}
impl LibraryEntry {
    fn resources(&self) -> Vec<&Resource> {
        match &self.content {
            LibraryContent::Project(p) => p.resources.iter().collect(),
            LibraryContent::Character(c) => c.resources.iter().collect(),
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
pub struct LibrarySession {
    document: LibraryDocument,
    store: Store<LibraryDocument>,
    sources: BTreeMap<BlobKey, File>,
}
impl LibrarySession {
    pub fn create(path: &Path) -> Result<Self, StorageError> {
        let document = LibraryDocument {
            schema_version: 1,
            id: format!("library_{}", uuid::Uuid::new_v4().simple()),
            revision: 0,
            entries: vec![],
        };
        let mut store = Store::select(path, "library.json")?;
        let mut sources = BTreeMap::new();
        store.save(&document, &mut sources)?;
        Ok(Self {
            document,
            store,
            sources,
        })
    }
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let (store, document) = Store::<LibraryDocument>::open(path, "library.json")?;
        let sources = store.files(&document)?;
        Ok(Self {
            document,
            store,
            sources,
        })
    }
    pub fn document(&self) -> &LibraryDocument {
        &self.document
    }
    /// Explicit save only. replace=(entry id, expected revision) updates a selected entry.
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
        candidate.entries.retain(|entry| entry.id != id);
        candidate.entries.push(LibraryEntry {
            id: id.clone(),
            revision,
            title,
            content,
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
        for (key, mut source) in self.select_sources(entry)? {
            sources.insert(key, staging.copy(&mut source)?);
        }
        for resource in &character.resources {
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
