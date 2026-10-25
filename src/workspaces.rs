//! Agent workspaces under the MCP root: one folder per workspace in its
//! creation date folder, reachable by agents only with the workspace key.
//! Agents never see these paths; the MCP proxy maps them to file IDs.
use crate::private_storage::Root;
use crate::workspace_ids as ids;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const KEY_FILE: &str = "AGENT-KEY.txt";
pub const META_FILE: &str = "workspace.json";
/// Uploaded images; audio and video may be larger.
pub const MAX_IMAGE_UPLOAD: u64 = 64 << 20;
pub const MAX_MEDIA_UPLOAD: u64 = 2 << 30;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    pub schema_version: u32,
    pub id: String,
    pub title: String,
    pub created_ms: u64,
    pub key_sha256: String,
}
#[derive(Clone, Debug)]
pub struct Workspace {
    pub meta: Meta,
    /// Absolute folder on the verified volume.
    pub folder: PathBuf,
    /// The folder relative to the MCP root (`2026-10-25/1432-title-k7q2`).
    pub relative: PathBuf,
}
impl Workspace {
    pub fn images(&self) -> PathBuf {
        self.folder.join("images")
    }
    pub fn videos(&self) -> PathBuf {
        self.folder.join("videos")
    }
    /// The summary the UI lists; never sent to agents.
    pub fn summary(&self) -> Value {
        json!({"workspace_id":self.meta.id,"title":self.meta.title,"created_ms":self.meta.created_ms,
            "path":self.folder,"folder":self.relative})
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct FileEntry {
    pub file_id: String,
    pub name: String,
    /// image, video, audio or other.
    pub kind: &'static str,
    /// upload or generated.
    pub origin: &'static str,
    /// The video project folder name for its outputs.
    pub project: Option<String>,
    pub mime_type: &'static str,
    pub bytes: u64,
    pub created_ms: u64,
    pub sha256: String,
    /// Inside the workspace, slash-separated.
    pub relative: String,
    #[serde(skip)]
    pub path: PathBuf,
}
#[derive(Debug)]
pub struct Error(pub &'static str);
impl From<&'static str> for Error {
    fn from(code: &'static str) -> Self {
        Self(code)
    }
}
type Result<T> = std::result::Result<T, Error>;

#[derive(Default)]
struct Index {
    by_key: HashMap<String, String>,
    by_id: HashMap<String, PathBuf>,
    /// SHA-256 by (path, size, modified) so listings hash each file once.
    digests: HashMap<(PathBuf, u64, u128), String>,
}
pub struct Workspaces {
    root: Root,
    path: PathBuf,
    index: Mutex<Index>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
fn random<const N: usize>() -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(|_| "workspace_random_unavailable")?;
    Ok(bytes)
}
/// A short, filesystem-safe form of a title.
pub fn slug(title: &str, max: usize) -> String {
    let mut out = String::new();
    for c in title.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= max {
            break;
        }
    }
    out.trim_end_matches('-').to_owned()
}
fn local_time() -> (String, String) {
    let date = imagine_video_contract::project::local_date();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&now, &mut tm) };
    (
        format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8]),
        format!("{:02}{:02}", tm.tm_hour, tm.tm_min),
    )
}
fn key_text(key: &str) -> String {
    format!(
        "{key}\n\nAgent key for this Imagine workspace. Give it to an agent that should\ncontinue this work; the agent opens the workspace with workspace_open.\n"
    )
}
/// What the first bytes of a file are, by content.
pub fn media_type(head: &[u8]) -> Option<(&'static str, &'static str, &'static str)> {
    let at = |offset: usize, magic: &[u8]| head.get(offset..offset + magic.len()) == Some(magic);
    Some(if at(0, b"\x89PNG\r\n\x1a\n") {
        ("image", "image/png", "png")
    } else if at(0, b"\xff\xd8\xff") {
        ("image", "image/jpeg", "jpg")
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        ("image", "image/webp", "webp")
    } else if at(0, b"RIFF") && at(8, b"WAVE") {
        ("audio", "audio/wav", "wav")
    } else if at(0, b"fLaC") {
        ("audio", "audio/flac", "flac")
    } else if at(0, b"OggS") {
        ("audio", "audio/ogg", "ogg")
    } else if at(0, b"ID3") || at(0, b"\xff\xfb") || at(0, b"\xff\xf3") || at(0, b"\xff\xf2") {
        ("audio", "audio/mpeg", "mp3")
    } else if at(4, b"ftypM4A") {
        ("audio", "audio/mp4", "m4a")
    } else if at(4, b"ftypqt") {
        ("video", "video/quicktime", "mov")
    } else if at(4, b"ftyp") {
        ("video", "video/mp4", "mp4")
    } else if at(0, b"\x1a\x45\xdf\xa3") {
        ("video", "video/webm", "webm")
    } else {
        return None;
    })
}
/// A safe file name for an upload, with the extension of its real type.
pub fn upload_name(requested: &str, extension: &str) -> String {
    let stem = Path::new(requested)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let stem = slug(stem, 60);
    format!(
        "{}.{extension}",
        if stem.is_empty() { "upload" } else { &stem }
    )
}

impl Workspaces {
    /// Workspaces in `root`, the verified MCP root.
    pub fn open(root: Root) -> Self {
        let this = Self {
            path: root.layout().root().to_path_buf(),
            root,
            index: Mutex::new(Index::default()),
        };
        this.scan();
        this
    }
    pub fn root(&self) -> &Path {
        &self.path
    }
    fn meta_at(&self, relative: &Path) -> Option<Meta> {
        let bytes = self.root.read(&relative.join(META_FILE)).ok()??;
        serde_json::from_slice::<Meta>(&bytes).ok()
    }
    /// Rebuilds the key and ID index from `<root>/<date>/<workspace>/workspace.json`.
    fn scan(&self) {
        let mut found = vec![];
        for date in std::fs::read_dir(self.root())
            .into_iter()
            .flatten()
            .flatten()
        {
            if !date.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            for folder in std::fs::read_dir(date.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                if !folder.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                let relative = PathBuf::from(date.file_name()).join(folder.file_name());
                if let Some(meta) = self.meta_at(&relative) {
                    found.push((meta, relative));
                }
            }
        }
        if let Ok(mut index) = self.index.lock() {
            index.by_key.clear();
            index.by_id.clear();
            for (meta, relative) in found {
                index
                    .by_key
                    .insert(meta.key_sha256.clone(), meta.id.clone());
                index.by_id.insert(meta.id, relative);
            }
        }
    }
    fn load(&self, id: &str) -> Result<Workspace> {
        let relative = self
            .index
            .lock()
            .map_err(|_| "workspace_unavailable")?
            .by_id
            .get(id)
            .cloned()
            .ok_or("workspace_not_found")?;
        let meta = self.meta_at(&relative).ok_or("workspace_not_found")?;
        if meta.id != id {
            return Err(Error("workspace_not_found"));
        }
        // Each access re-proves the folder is a real directory on the volume.
        self.root
            .directory(&relative)
            .map_err(|_| "workspace_unavailable")?;
        Ok(Workspace {
            folder: self.root().join(&relative),
            relative,
            meta,
        })
    }
    pub fn create(&self, title: &str) -> Result<(Workspace, String)> {
        let title: String = title.trim().chars().take(80).collect();
        let key = ids::new_key(&random::<17>()?);
        let id = format!("ws_{}", uuid::Uuid::new_v4().simple());
        let (date, time) = local_time();
        let mut name = time;
        let slug = slug(&title, 40);
        if !slug.is_empty() {
            name.push('-');
            name.push_str(&slug);
        }
        name.push('-');
        name.push_str(&id[3..7]);
        let relative = PathBuf::from(date).join(name);
        for child in ["uploads", "images", "videos"] {
            self.root.directory(&relative.join(child))?;
        }
        let meta = Meta {
            schema_version: 1,
            id: id.clone(),
            title,
            created_ms: now_ms(),
            key_sha256: ids::key_hash(&key),
        };
        self.root
            .write(&relative.join(KEY_FILE), key_text(&key).as_bytes())?;
        self.root.write(
            &relative.join(META_FILE),
            &serde_json::to_vec_pretty(&meta).map_err(|_| "workspace_invalid")?,
        )?;
        if let Ok(mut index) = self.index.lock() {
            index.by_key.insert(meta.key_sha256.clone(), id.clone());
            index.by_id.insert(id.clone(), relative);
        }
        Ok((self.load(&id)?, key))
    }
    /// The workspace for `key`; unknown keys say nothing about others.
    pub fn resolve(&self, key: &str) -> Result<Workspace> {
        if !ids::key_shaped(key) {
            return Err(Error("workspace_unknown"));
        }
        let hash = ids::key_hash(key);
        let lookup = |this: &Self| {
            this.index
                .lock()
                .ok()
                .and_then(|index| index.by_key.get(&hash).cloned())
        };
        let id = match lookup(self) {
            Some(id) => id,
            None => {
                self.scan();
                lookup(self).ok_or("workspace_unknown")?
            }
        };
        let workspace = self.load(&id).map_err(|_| "workspace_unknown")?;
        if workspace.meta.key_sha256 != hash {
            return Err(Error("workspace_unknown"));
        }
        Ok(workspace)
    }
    pub fn get(&self, id: &str) -> Result<Workspace> {
        self.load(id).or_else(|_| {
            self.scan();
            self.load(id)
        })
    }
    /// All workspaces, newest first, for the UI.
    pub fn list(&self) -> Vec<Workspace> {
        self.scan();
        let ids: Vec<String> = self
            .index
            .lock()
            .map(|index| index.by_id.keys().cloned().collect())
            .unwrap_or_default();
        let mut all: Vec<_> = ids.iter().filter_map(|id| self.load(id).ok()).collect();
        all.sort_by_key(|w| std::cmp::Reverse(w.meta.created_ms));
        all
    }
    pub fn key(&self, id: &str) -> Result<String> {
        let workspace = self.get(id)?;
        let bytes = self
            .root
            .read(&workspace.relative.join(KEY_FILE))?
            .ok_or("workspace_key_missing")?;
        let key = String::from_utf8_lossy(&bytes)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_owned();
        if ids::key_hash(&key) != workspace.meta.key_sha256 {
            return Err(Error("workspace_key_missing"));
        }
        Ok(key)
    }
    /// A new key; the old one stops working at once.
    pub fn replace_key(&self, id: &str) -> Result<String> {
        let mut workspace = self.get(id)?;
        let key = ids::new_key(&random::<17>()?);
        workspace.meta.key_sha256 = ids::key_hash(&key);
        self.root.write(
            &workspace.relative.join(META_FILE),
            &serde_json::to_vec_pretty(&workspace.meta).map_err(|_| "workspace_invalid")?,
        )?;
        self.root.write(
            &workspace.relative.join(KEY_FILE),
            key_text(&key).as_bytes(),
        )?;
        self.scan();
        Ok(key)
    }
    /// Whether `path` is a workspace's image output folder.
    pub fn is_image_folder(&self, path: &Path) -> bool {
        path.file_name() == Some("images".as_ref())
            && path
                .parent()
                .and_then(|folder| folder.strip_prefix(self.root()).ok())
                .and_then(|relative| self.meta_at(relative))
                .is_some()
    }
    /// The workspace and the inside-relative path of an absolute `path`.
    pub fn locate(&self, workspace: &Workspace, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&workspace.folder).ok()?;
        let parts: Vec<_> = relative
            .components()
            .map(|c| c.as_os_str().to_str())
            .collect::<Option<_>>()?;
        Some(parts.join("/"))
    }
    fn entry(&self, workspace: &Workspace, path: &Path) -> Option<FileEntry> {
        let relative = self.locate(workspace, path)?;
        let metadata = std::fs::symlink_metadata(path).ok()?;
        let name = path.file_name()?.to_str()?.to_owned();
        if !metadata.is_file() || name.starts_with('.') {
            return None;
        }
        let mut head = [0u8; 16];
        let read = std::fs::File::open(path)
            .and_then(|mut f| f.read(&mut head))
            .ok()?;
        let (kind, mime_type) = media_type(&head[..read])
            .map(|(kind, mime, _)| (kind, mime))
            .unwrap_or(("other", "application/octet-stream"));
        let parts: Vec<&str> = relative.split('/').collect();
        let (origin, project) = match parts.as_slice() {
            ["uploads", ..] => ("upload", None),
            ["videos", project, ..] => ("generated", Some((*project).to_owned())),
            _ => ("generated", None),
        };
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_millis());
        let digest_key = (path.to_path_buf(), metadata.len(), modified);
        let cached = self
            .index
            .lock()
            .ok()
            .and_then(|index| index.digests.get(&digest_key).cloned());
        let sha256 = match cached {
            Some(digest) => digest,
            None => {
                use sha2::Digest;
                let mut hasher = sha2::Sha256::new();
                let mut file = std::fs::File::open(path).ok()?;
                std::io::copy(&mut file, &mut hasher).ok()?;
                let digest = format!("{:x}", hasher.finalize());
                if let Ok(mut index) = self.index.lock() {
                    index.digests.insert(digest_key, digest.clone());
                }
                digest
            }
        };
        Some(FileEntry {
            file_id: ids::file_id(&workspace.meta.id, &relative),
            name,
            kind,
            origin,
            project,
            mime_type,
            bytes: metadata.len(),
            created_ms: modified as u64,
            sha256,
            relative,
            path: path.to_path_buf(),
        })
    }
    /// Uploads, generated images and video project outputs, oldest first.
    pub fn files(&self, workspace: &Workspace) -> Vec<FileEntry> {
        let mut folders = vec![workspace.folder.join("uploads"), workspace.images()];
        for project in std::fs::read_dir(workspace.videos())
            .into_iter()
            .flatten()
            .flatten()
        {
            if project.file_type().is_ok_and(|t| t.is_dir()) {
                for class in ["images", "videos", "audio"] {
                    folders.push(project.path().join("outputs").join(class));
                }
            }
        }
        let mut files = vec![];
        while let Some(folder) = folders.pop() {
            for entry in std::fs::read_dir(&folder).into_iter().flatten().flatten() {
                let Ok(kind) = entry.file_type() else {
                    continue;
                };
                if kind.is_dir() {
                    folders.push(entry.path());
                } else if let Some(file) = self.entry(workspace, &entry.path()) {
                    files.push(file);
                }
            }
        }
        files.sort_by(|a, b| (a.created_ms, &a.relative).cmp(&(b.created_ms, &b.relative)));
        files
    }
    pub fn file(&self, workspace: &Workspace, file_id: &str) -> Result<FileEntry> {
        self.files(workspace)
            .into_iter()
            .find(|f| f.file_id == file_id)
            .ok_or(Error("file_not_found"))
    }
    /// The entry for a file Imagine wrote into the workspace.
    pub fn file_at(&self, workspace: &Workspace, path: &Path) -> Option<FileEntry> {
        self.entry(workspace, path)
    }
    /// A new upload named after `requested`, unique in `uploads/`.
    pub fn upload(
        &self,
        workspace: &Workspace,
        requested: &str,
        head: &[u8],
    ) -> Result<(crate::private_storage::NewFile, String, u64)> {
        let (kind, _, extension) = media_type(head).ok_or("upload_type_unsupported")?;
        let limit = if kind == "image" {
            MAX_IMAGE_UPLOAD
        } else {
            MAX_MEDIA_UPLOAD
        };
        let base = upload_name(requested, extension);
        let uploads = workspace.folder.join("uploads");
        let stem = base
            .rsplit_once('.')
            .map_or(base.as_str(), |(stem, _)| stem);
        let name = (1..10_000)
            .map(|n| {
                if n == 1 {
                    base.clone()
                } else {
                    format!("{stem}-{n}.{extension}")
                }
            })
            .find(|name| std::fs::symlink_metadata(uploads.join(name)).is_err())
            .ok_or("upload_name_unavailable")?;
        // Publishing never replaces a file that appears meanwhile.
        Ok((
            self.root
                .create_new(&workspace.relative.join("uploads").join(&name))?,
            name,
            limit,
        ))
    }
    /// A new, empty folder for a video project: `videos/<slug>`.
    pub fn project_folder(&self, workspace: &Workspace, title: &str) -> Result<PathBuf> {
        let base = {
            let s = slug(title, 40);
            if s.is_empty() { "video".to_owned() } else { s }
        };
        for attempt in 1..1000 {
            let name = if attempt == 1 {
                base.clone()
            } else {
                format!("{base}-{attempt}")
            };
            let path = workspace.videos().join(&name);
            if std::fs::symlink_metadata(&path).is_err() {
                self.root
                    .directory(&workspace.relative.join("videos").join(&name))?;
                return Ok(path);
            }
        }
        Err(Error("workspace_project_folder_unavailable"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_slugs_and_media_types() {
        assert_eq!(slug("Forest Chase: Part 2!", 40), "forest-chase-part-2");
        assert_eq!(slug("  ", 40), "");
        assert_eq!(upload_name("My Ref (1).PNG", "png"), "my-ref-1.png");
        assert_eq!(upload_name("", "jpg"), "upload.jpg");
        assert_eq!(
            media_type(b"\x89PNG\r\n\x1a\n...."),
            Some(("image", "image/png", "png"))
        );
        assert_eq!(media_type(b"RIFF\0\0\0\0WEBPVP8 ").unwrap().0, "image");
        assert_eq!(media_type(b"\0\0\0\x18ftypisom").unwrap().0, "video");
        assert_eq!(media_type(b"RIFF\0\0\0\0WAVEfmt ").unwrap().0, "audio");
        assert_eq!(media_type(b"#!/bin/sh"), None);
    }
}
