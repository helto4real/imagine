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
    pub session: std::sync::Arc<crate::session::Session>,
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
              output TEXT, error TEXT, UNIQUE(owner,idempotency_key));
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
        Ok(Self { db, media, session })
    }
    pub async fn close(&self) -> Result<()> {
        self.db.close().await;
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
            let req: Value = serde_json::from_str(r.get("request"))?;
            let output: Option<String> = r.get("output");
            let error: Option<String> = r.get("error");
            let started: Option<i64> = r.get("started"); let finished: Option<i64> = r.get("finished");
            Ok(json!({"id":id,"status":r.get::<String,_>("status"),"created_at_ms":r.get::<i64,_>("created"),
                "started_at_ms":started,"finished_at_ms":finished,
                "execution_to_saved_ms":started.zip(finished).map(|(s,f)|f-s),
                "effective":req,"output":output.map(|s|serde_json::from_str::<Value>(&s)).transpose()?,
                "error":error.map(|code|json!({"code":code}))}))
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
        if self.asset(id, owner).await?.is_none() {
            bail!("asset_not_found");
        }
        // Only IDs obtained from our database reach the path, never user paths.
        Ok(tokio::fs::read(self.media.join(id)).await?)
    }
    pub async fn save_asset(&self, owner: &str, bytes: Vec<u8>) -> Result<Value> {
        let info = tokio::task::spawn_blocking(move || inspect_image(bytes)).await??;
        let asset_id = id("asset");
        let tmp = self.media.join(format!("{asset_id}.partial"));
        let mut f = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .await?;
        use tokio::io::AsyncWriteExt;
        f.write_all(&info.bytes).await?;
        f.sync_all().await?;
        drop(f);
        tokio::fs::rename(&tmp, self.media.join(&asset_id)).await?;
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
