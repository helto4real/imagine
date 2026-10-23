use anyhow::{Context, Result, bail};
use image::ImageReader;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
pub struct Store {
    pub db: SqlitePool,
    pub media: PathBuf,
    pub videos: crate::video::output::Outputs,
    media_files: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, std::fs::File>>>,
    uploads:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<[u8]>>>>,
    pub session: std::sync::Arc<crate::session::Session>,
}
struct VideoPublication {
    delivery: Option<Box<crate::video::assembly::Plan>>,
    asset_id: String,
    take: Option<Value>,
    owned: Option<std::fs::File>,
}
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
pub fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}
pub fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
impl Store {
    pub async fn open(data: &Path) -> Result<Self> {
        let session = crate::session::Session::new(data)?;
        let db = SqlitePoolOptions::new()
            // Retiring the sole SQLite connection would silently destroy the RAM database.
            .min_connections(1)
            .max_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_with(
                SqliteConnectOptions::new()
                    .in_memory(true)
                    .pragma("temp_store", "MEMORY")
                    .foreign_keys(true),
            )
            .await?;
        sqlx::raw_sql("
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO settings VALUES ('paused','false');
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
              request TEXT NOT NULL, request_hash TEXT NOT NULL, idempotency_key TEXT,
              created INTEGER NOT NULL, started INTEGER, finished INTEGER,
              output TEXT, error TEXT, error_details TEXT, UNIQUE(owner,idempotency_key));
            CREATE TABLE IF NOT EXISTS job_batches (
              owner TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL,
              job_ids TEXT NOT NULL, PRIMARY KEY(owner,idempotency_key));
            CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, mime TEXT NOT NULL,
              width INTEGER NOT NULL, height INTEGER NOT NULL, bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL, created INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (
              id TEXT PRIMARY KEY, status TEXT NOT NULL, mode TEXT NOT NULL, created INTEGER NOT NULL, finished INTEGER);
        ").execute(&db).await?;
        let media = session.media.clone();
        Ok(Self {
            db,
            media,
            session,
            uploads: Default::default(),
            media_files: Default::default(),
            videos: crate::video::output::Outputs::configure(&data.join("video-output.json")),
        })
    }
    pub async fn close(&self) -> Result<()> {
        self.media_files
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .clear();
        self.db.close().await;
        self.uploads
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .clear();
        self.videos
            .published
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .clear();
        self.session.clear()
    }
    pub async fn paused(&self) -> Result<bool> {
        Ok(
            sqlx::query_scalar::<_, String>("SELECT value FROM settings WHERE key='paused'")
                .fetch_one(&self.db)
                .await?
                == "true",
        )
    }
    pub async fn set_paused(&self, value: bool) -> Result<()> {
        sqlx::query("UPDATE settings SET value=? WHERE key='paused'")
            .bind(value.to_string())
            .execute(&self.db)
            .await?;
        Ok(())
    }
    pub async fn job(&self, id: &str, owner: &str) -> Result<Option<Value>> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id=? AND owner=?")
            .bind(id)
            .bind(owner)
            .fetch_optional(&self.db)
            .await?;
        row.map(|r| -> Result<Value> {
            let mut req: Value = serde_json::from_str(r.get("request"))?;
            req.as_object_mut().unwrap().remove("resolved_models");
            let output: Option<String> = r.get("output");
            let error: Option<String> = r.get("error");
            let details: Option<String> = r.get("error_details");
            let error = error.map(|code| {
                let mut value = json!({"code":code});
                if code == "model_component_load_failed" && let Some(details) = details
                    .and_then(|text|serde_json::from_str::<crate::component_error::ComponentLoadError>(&text).ok())
                    .filter(crate::component_error::ComponentLoadError::valid) {
                    value.as_object_mut().unwrap().extend(serde_json::to_value(details).unwrap().as_object().unwrap().clone());
                }
                value
            });
            let started: Option<i64> = r.get("started"); let finished: Option<i64> = r.get("finished");
            Ok(json!({"id":id,"status":r.get::<String,_>("status"),"created_at_ms":r.get::<i64,_>("created"),
                "started_at_ms":started,"finished_at_ms":finished,
                "execution_to_saved_ms":started.zip(finished).map(|(s,f)|f-s),
                "effective":req,"output":output.map(|s|serde_json::from_str::<Value>(&s)).transpose()?,
                "error":error}))
        }).transpose()
    }
    pub async fn operation(&self, id: &str) -> Result<Option<Value>> {
        Ok(sqlx::query("SELECT * FROM operations WHERE id=?").bind(id).fetch_optional(&self.db).await?.map(|r|
            json!({"id":id,"status":r.get::<String,_>("status"),"active_job":r.get::<String,_>("mode"),
              "created_at_ms":r.get::<i64,_>("created"),"finished_at_ms":r.get::<Option<i64>,_>("finished")})))
    }
    pub async fn asset(&self, id: &str, owner: &str) -> Result<Option<Value>> {
        Ok(sqlx::query("SELECT * FROM assets WHERE id=? AND owner=?").bind(id).bind(owner).fetch_optional(&self.db).await?.map(|r|
            json!({"id":id,"mime_type":r.get::<String,_>("mime"),"width":r.get::<i64,_>("width"),"height":r.get::<i64,_>("height"),
                "bytes":r.get::<i64,_>("bytes"),"sha256":r.get::<String,_>("sha256")})))
    }
    pub async fn read_asset(&self, id: &str, owner: &str) -> Result<Vec<u8>> {
        let info = self.asset(id, owner).await?.context("asset_not_found")?;
        let bytes = if let Some(mut file) = self.open_media(id)? {
            use std::io::Read;
            let mut bytes = vec![];
            file.read_to_end(&mut bytes)?;
            bytes
        } else {
            self.upload_bytes(id)?
                .map(|b| b.to_vec())
                .context("asset_not_found")?
        };
        if info["bytes"].as_u64() != Some(bytes.len() as u64)
            || info["sha256"].as_str() != Some(hash(&bytes).as_str())
        {
            bail!(crate::engine::EngineError("asset_content_changed"));
        }
        Ok(bytes)
    }

    pub fn upload_bytes(&self, id: &str) -> Result<Option<std::sync::Arc<[u8]>>> {
        Ok(self
            .uploads
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .get(id)
            .cloned())
    }
    pub fn open_media(&self, id: &str) -> Result<Option<std::fs::File>> {
        use std::os::fd::AsRawFd;
        let owned = self
            .media_files
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .get(id)
            .map(|file| std::fs::File::open(format!("/proc/self/fd/{}", file.as_raw_fd())))
            .transpose()
            .map_err(anyhow::Error::from)?;
        if owned.is_some() {
            return Ok(owned);
        }
        self.videos.open(id).map_err(anyhow::Error::msg)
    }
    pub async fn register_media(
        &self,
        owner: &str,
        file: std::fs::File,
        info: &crate::video::media::Info,
    ) -> Result<Value> {
        Ok(self
            .register_media_batch(owner, vec![(file, info.clone())])
            .await?
            .remove(0))
    }
    pub async fn register_media_batch(
        &self,
        owner: &str,
        files: Vec<(std::fs::File, crate::video::media::Info)>,
    ) -> Result<Vec<Value>> {
        let prepared = tokio::task::spawn_blocking(move || -> Result<Vec<_>> {
            files.into_iter().map(|(mut file, info)| {
                let (digest, bytes) = crate::video::bundle::digest(&mut file)?;
                let asset_id = id("asset");
                let metadata = json!({"id":asset_id,"mime_type":info.mime_type,"width":info.width,"height":info.height,"bytes":bytes,"sha256":digest,"media":info});
                Ok((asset_id, file, metadata))
            }).collect()
        }).await??;
        let mut tx = self.db.begin().await?;
        for (asset_id, _, info) in &prepared {
            sqlx::query("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)")
                .bind(asset_id)
                .bind(owner)
                .bind(info["mime_type"].as_str().unwrap())
                .bind(info["width"].as_u64().unwrap() as i64)
                .bind(info["height"].as_u64().unwrap() as i64)
                .bind(info["bytes"].as_u64().unwrap() as i64)
                .bind(info["sha256"].as_str().unwrap())
                .bind(now())
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        let mut handles = self
            .media_files
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?;
        let mut result = Vec::with_capacity(prepared.len());
        for (id, file, value) in prepared {
            handles.insert(id, file);
            result.push(value);
        }
        Ok(result)
    }
    pub async fn reserve_images(
        &self,
        request: &crate::contract::JobRequest,
        owner: &str,
    ) -> Result<Vec<crate::video::output::Pending>> {
        let destination = self
            .videos
            .destination()
            .map_err(crate::engine::EngineError)?;
        let source = if request
            .resolved_profile
            .as_ref()
            .is_some_and(|p| p.id == "image_upscale")
        {
            if let Some(id) = request.inputs.reference_images.first() {
                if self.asset(id, owner).await?.is_none() {
                    bail!(crate::engine::EngineError("asset_not_found"));
                }
                self.videos
                    .published
                    .lock()
                    .map_err(|_| anyhow::anyhow!("storage_error"))?
                    .get(id)
                    .cloned()
            } else {
                None
            }
        } else {
            None
        };
        let count = if request.operation == crate::contract::Operation::CharacterSheet {
            3
        } else {
            1
        };
        (0..count)
            .map(|_| {
                destination
                    .reserve_image(
                        &request.client_metadata["video_destination"],
                        32 * 1024 * 1024 * count,
                        source.as_ref(),
                    )
                    .map_err(|e| crate::engine::EngineError(e).into())
            })
            .collect()
    }
    /// Publish all roles and job success together. Pending files roll back until commit.
    pub async fn publish_output(
        &self,
        job: &str,
        owner: &str,
        generated: impl Into<crate::generation::Generated>,
    ) -> Result<()> {
        match generated.into() {
            crate::generation::Generated::Delivery {
                bytes,
                plan,
                pending,
            } => {
                self.publish_video(
                    job,
                    owner,
                    &bytes,
                    *pending,
                    VideoPublication {
                        delivery: Some(plan),
                        asset_id: id("asset"),
                        take: None,
                        owned: None,
                    },
                )
                .await
            }
            crate::generation::Generated::Video { bytes, pending, .. } => {
                self.publish_video(
                    job,
                    owner,
                    &bytes,
                    *pending,
                    VideoPublication {
                        delivery: None,
                        asset_id: id("asset"),
                        take: None,
                        owned: None,
                    },
                )
                .await
            }
            crate::generation::Generated::Images { artifacts, pending } => {
                self.publish_images(job, owner, artifacts, pending).await
            }
        }
    }
    pub(crate) async fn publish_scene(
        &self,
        job: &str,
        owner: &str,
        generated: crate::generation::Generated,
        take: Value,
        owned: std::fs::File,
    ) -> Result<String> {
        let crate::generation::Generated::Video { bytes, pending, .. } = generated else {
            bail!("invalid_scene_output");
        };
        let asset = id("asset");
        self.publish_video(
            job,
            owner,
            &bytes,
            *pending,
            VideoPublication {
                delivery: None,
                asset_id: asset.clone(),
                take: Some(take),
                owned: Some(owned),
            },
        )
        .await?;
        Ok(asset)
    }
    pub(crate) async fn publish_delivery(
        &self,
        job: &str,
        owner: &str,
        generated: crate::generation::Generated,
        asset_id: String,
        owned: std::fs::File,
    ) -> Result<()> {
        let crate::generation::Generated::Delivery {
            bytes,
            pending,
            plan,
        } = generated
        else {
            bail!("invalid_delivery_output");
        };
        self.publish_video(
            job,
            owner,
            &bytes,
            *pending,
            VideoPublication {
                delivery: Some(plan),
                asset_id,
                take: None,
                owned: Some(owned),
            },
        )
        .await
    }
    async fn publish_images(
        &self,
        job: &str,
        owner: &str,
        artifacts: Vec<crate::generation::Artifact>,
        mut pending: Vec<crate::video::output::Pending>,
    ) -> Result<()> {
        if artifacts.is_empty() || pending.len() != artifacts.len() {
            bail!("image_output_not_reserved");
        }
        let mut inspected = Vec::new();
        for a in artifacts {
            inspected.push((
                a.role,
                tokio::task::spawn_blocking(move || inspect_image(a.bytes)).await??,
            ));
        }
        let mut tx = self.db.begin().await?;
        let mut published = Vec::new();
        let mut assets = Vec::new();
        let mut group = None;
        for ((role, info), pending) in inspected.into_iter().zip(pending.iter_mut()) {
            let saved = pending
                .publish_image(&info.bytes, role, &info.mime, group.as_deref())
                .map_err(crate::engine::EngineError)?;
            if role == "character_sheet" {
                group = saved
                    .path
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .and_then(|v| v.strip_suffix("_character_sheet"))
                    .map(str::to_owned);
            }
            let id = id("asset");
            let digest = hash(&info.bytes);
            sqlx::query("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)")
                .bind(&id)
                .bind(owner)
                .bind(&info.mime)
                .bind(info.width)
                .bind(info.height)
                .bind(info.bytes.len() as i64)
                .bind(&digest)
                .bind(now())
                .execute(&mut *tx)
                .await?;
            assets.push(json!({"id":id,"role":role,"mime_type":info.mime,"width":info.width,"height":info.height,"bytes":info.bytes.len(),"sha256":digest,"saved_path":saved.path,"persistent":true}));
            published.push((id, saved));
        }
        let output = json!({"assets":assets,"format":"png","persistent":true,"saved_path":assets[0]["saved_path"]});
        let updated = sqlx::query("UPDATE jobs SET status='succeeded',output=?,finished=? WHERE id=? AND owner=? AND status='running'").bind(output.to_string()).bind(now()).bind(job).bind(owner).execute(&mut *tx).await?;
        if updated.rows_affected() != 1 {
            bail!("output job is no longer running");
        }
        {
            let mut map = self
                .videos
                .published
                .lock()
                .map_err(|_| anyhow::anyhow!("storage_error"))?;
            for (id, saved) in &published {
                map.insert(id.clone(), saved.clone());
            }
        }
        if let Err(error) = tx.commit().await {
            let mut map = self
                .videos
                .published
                .lock()
                .map_err(|_| anyhow::anyhow!("storage_error"))?;
            for (id, _) in published {
                map.remove(&id);
            }
            return Err(error.into());
        }
        for pending in pending {
            pending.commit();
        }
        Ok(())
    }
    async fn publish_video(
        &self,
        job: &str,
        owner: &str,
        bytes: &[u8],
        mut pending: crate::video::output::Pending,
        publication: VideoPublication,
    ) -> Result<()> {
        let VideoPublication {
            delivery,
            asset_id,
            take,
            owned,
        } = publication;
        let (width, height, video) = if let Some(plan) = &delivery {
            (
                plan.delivery.width,
                plan.delivery.height,
                json!({"frames":plan.frames,"fps":crate::video::assembly::seconds(plan.delivery.fps),"fps_rational":plan.delivery.fps,"duration_seconds":plan.frames as f64/crate::video::assembly::seconds(plan.delivery.fps),"audio":true,"sample_rate":plan.sample_rate,"channels":2,"samples":plan.samples}),
            )
        } else {
            let decoded = crate::short_av::decode(bytes).map_err(anyhow::Error::msg)?;
            let shape = decoded.shape;
            (
                shape.width,
                shape.height,
                json!({"frames":shape.frames,"fps":shape.fps,"duration_seconds":shape.seconds(),"audio":true,"sample_rate":32000,"channels":2,"samples":shape.audio_samples()}),
            )
        };
        let published = pending.publish(bytes).map_err(crate::engine::EngineError)?;
        let digest = hash(bytes);
        let mut output = json!({"format":"mp4","saved_path":published.path,"persistent":true,"assets":[{"id":asset_id,"role":"video","mime_type":crate::short_av::MIME,"width":width,"height":height,"bytes":bytes.len(),"sha256":digest}],"video":video});
        if let Some(plan) = delivery {
            output["delivery"] = json!(plan);
        }
        if let Some(take) = take {
            output["take"] = take;
        }
        let mut tx = self.db.begin().await?;
        sqlx::query("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)")
            .bind(&asset_id)
            .bind(owner)
            .bind(crate::short_av::MIME)
            .bind(width)
            .bind(height)
            .bind(bytes.len() as i64)
            .bind(digest)
            .bind(now())
            .execute(&mut *tx)
            .await?;
        let updated=sqlx::query("UPDATE jobs SET status='succeeded',output=?,finished=? WHERE id=? AND owner=? AND status='running'").bind(output.to_string()).bind(now()).bind(job).bind(owner).execute(&mut *tx).await?;
        if updated.rows_affected() != 1 {
            bail!("output job is no longer running");
        }
        self.videos
            .published
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .insert(asset_id.clone(), published);
        if let Some(file) = owned {
            self.media_files
                .lock()
                .map_err(|_| anyhow::anyhow!("storage_error"))?
                .insert(asset_id.clone(), file);
        }
        if let Err(e) = tx.commit().await {
            self.videos.published.lock().unwrap().remove(&asset_id);
            self.media_files.lock().unwrap().remove(&asset_id);
            return Err(e.into());
        }
        pending.commit();
        Ok(())
    }
    pub async fn save_asset(&self, owner: &str, bytes: Vec<u8>) -> Result<Value> {
        let info = tokio::task::spawn_blocking(move || inspect_image(bytes)).await??;
        let asset_id = id("asset");
        sqlx::query("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)")
            .bind(&asset_id)
            .bind(owner)
            .bind(&info.mime)
            .bind(info.width)
            .bind(info.height)
            .bind(info.bytes.len() as i64)
            .bind(hash(&info.bytes))
            .bind(now())
            .execute(&self.db)
            .await?;
        self.uploads
            .lock()
            .map_err(|_| anyhow::anyhow!("storage_error"))?
            .insert(asset_id.clone(), info.bytes.into());
        self.asset(&asset_id, owner)
            .await?
            .context("saved asset missing")
    }
}
pub struct ImageInfo {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub mime: String,
}
pub fn inspect_image(bytes: Vec<u8>) -> Result<ImageInfo> {
    if bytes.len() > 32 * 1024 * 1024 {
        bail!("image too large");
    }
    let mut r = ImageReader::new(Cursor::new(&bytes)).with_guessed_format()?;
    let format = r.format().context("unknown image format")?;
    let mime = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        _ => bail!("unsupported image"),
    };
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(256 * 1024 * 1024);
    r.limits(limits);
    let img = r.decode()?;
    if u64::from(img.width()) * u64::from(img.height()) > 16_777_216 {
        bail!("image exceeds 16 megapixels");
    }
    Ok(ImageInfo {
        width: img.width(),
        height: img.height(),
        mime: mime.into(),
        bytes,
    })
}
