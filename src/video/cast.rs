//! Copy library characters and their media into a project's own resource transaction.
use super::{contract::*, persistence::*, service::Service};
use imagine_video_contract::creative::*;
use std::collections::BTreeMap;
pub(crate) struct Prepared {
    pub assets: Vec<AssetId>,
    pub files: Vec<(std::fs::File, super::media::Info)>,
    project_id: ProjectId,
    update: PreparedUpdate,
}
impl Prepared {
    pub fn commit(
        self,
        service: &mut Service,
        values: Vec<serde_json::Value>,
    ) -> Result<(), StorageError> {
        let session = service.projects.session_mut(&self.project_id)?;
        session.commit_prepared(self.update);
        let saved = session.autosave();
        let bindings = service.runtime_assets.entry(self.project_id).or_default();
        for (asset, value) in self.assets.into_iter().zip(values) {
            bindings.insert(asset, value["id"].as_str().unwrap().into());
        }
        saved
    }
}
pub(crate) fn from_library(
    service: &mut Service,
    library_id: &str,
    entry_id: &str,
    entry_revision: u64,
    project_id: &ProjectId,
    project_revision: u64,
    staging: &SessionStaging,
) -> Result<Prepared, StorageError> {
    let copy = service
        .libraries
        .get(library_id)
        .ok_or_else(|| StorageError::new("library_not_found"))?
        .copy_character(entry_id, entry_revision, staging)?;
    let session = service.projects.session_mut(project_id)?;
    if session.document().project.project_revision != project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    let mut document = session.document().clone();
    let mut ids = BTreeMap::new();
    let mut assets = vec![];
    let mut files = vec![];
    for resource in &copy.character.resources {
        let file = copy
            .sources
            .get(&resource.key())
            .ok_or_else(|| StorageError::new("resource_missing"))?;
        let info = super::media::detect(file).map_err(StorageError::new)?;
        let asset = AssetId(crate::storage::id("asset"));
        let mut owned = resource.clone();
        owned.id = crate::storage::id("reference_media");
        owned.kind = ResourceKind::Reference {
            asset_id: asset.clone(),
            revision: 1,
        };
        document.project.assets.insert(
            asset.clone(),
            Asset {
                library_source: None,
                name: String::new(),
                id: asset.clone(),
                revision: 1,
                kind: info.kind.clone(),
            },
        );
        document.resources.push(owned);
        assets.push(asset.clone());
        files.push((
            file.try_clone()
                .map_err(|_| StorageError::new("storage_io_failed"))?,
            info,
        ));
        ids.insert(resource.id.clone(), asset);
    }
    let id = crate::storage::id("reference");
    let images = copy
        .character
        .visual_ids
        .iter()
        .enumerate()
        .map(|(i, key)| {
            Ok(ReferenceImage {
                id: crate::storage::id("reference_image"),
                asset: ids
                    .get(key)
                    .cloned()
                    .ok_or_else(|| StorageError::new("character_resource_missing"))?,
                enabled: true,
                preview: i == 0,
                contribution: String::new(),
            })
        })
        .collect::<Result<Vec<_>, StorageError>>()?;
    let voice = copy
        .character
        .voice
        .map(|v| {
            Ok::<_, StorageError>(Voice {
                reference_settings: Default::default(),
                language: v.language,
                description: v.description,
                assets: v
                    .reference_ids
                    .iter()
                    .map(|key| {
                        ids.get(key)
                            .cloned()
                            .ok_or_else(|| StorageError::new("character_resource_missing"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .transpose()?;
    let mut reference = if let Some(mut reference) = copy.character.reference {
        let previous_id = reference.id.clone();
        for image in &mut reference.images {
            image.id = crate::storage::id("reference_image");
            image.asset = ids
                .get(&image.asset.0)
                .cloned()
                .ok_or_else(|| StorageError::new("character_resource_missing"))?;
        }
        if let Some(voice) = &mut reference.voice {
            let mut settings = BTreeMap::new();
            for asset in &mut voice.assets {
                let mapped = ids
                    .get(&asset.0)
                    .cloned()
                    .ok_or_else(|| StorageError::new("character_resource_missing"))?;
                if let Some(value) = voice.reference_settings.get(asset) {
                    settings.insert(mapped.clone(), value.clone());
                }
                *asset = mapped;
            }
            voice.reference_settings = settings;
        }
        if let Some(relationship) = &mut reference.relationship
            && relationship.target_reference_id.as_deref() == Some(previous_id.as_str())
        {
            relationship.target_reference_id = Some(id.clone());
        }
        reference
    } else {
        VisualReference {
            id: id.clone(),
            name: String::new(),
            alias: String::new(),
            kind: "Character".into(),
            description: String::new(),
            tags: vec![],
            enabled: true,
            strength: 1.,
            retention: String::new(),
            relationship: None,
            images,
            voice,
            origin: None,
        }
    };
    reference.id = id;
    reference.name = copy.character.name;
    reference.alias = copy.character.alias;
    reference.description = copy.character.description;
    reference.tags = copy.character.tags;
    reference.origin = Some(Origin {
        library_id: copy.origin.library_id,
        entry_id: copy.origin.entry_id,
        revision: copy.origin.entry_revision,
    });
    document.project.creative.references.push(reference);
    super::review::invalidate_changed(&session.document().project, &mut document)?;
    let update = session.prepare_update(project_revision, document, copy.sources, staging)?;
    Ok(Prepared {
        assets,
        files,
        project_id: project_id.clone(),
        update,
    })
}
