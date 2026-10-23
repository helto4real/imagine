//! Optional discovery routes. Catalog failures never enter generation validation.
use super::super::lora_catalog as catalog;
use super::*;

pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/loras", get(list))
        .route("/v1/video/loras/info", post(info))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InfoRequest {
    schema_version: u32,
    file: String,
}

fn root(app: &App) -> Result<PathBuf, Error> {
    let root = app
        .library
        .lock()
        .map(|library| library.snapshot().settings.root)
        .map_err(|_| Error::new(StatusCode::INTERNAL_SERVER_ERROR, "lora_catalog_internal"))?;
    if root.as_os_str().is_empty() {
        return Err(catalog_error(catalog::Error::CatalogUnavailable));
    }
    Ok(root.join("loras"))
}

fn catalog_error(error: catalog::Error) -> Error {
    let status = match error {
        catalog::Error::InfoMissing => StatusCode::NOT_FOUND,
        catalog::Error::InfoFormat => StatusCode::UNPROCESSABLE_ENTITY,
        catalog::Error::CatalogUnavailable | catalog::Error::InfoUnreadable => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
    };
    Error::new(status, error.code())
}

async fn list(State(app): State<App>) -> Result<Json<catalog::Catalog>, Error> {
    let root = root(&app)?;
    tokio::task::spawn_blocking(move || catalog::scan(&root))
        .await
        .map_err(|_| Error::new(StatusCode::INTERNAL_SERVER_ERROR, "lora_catalog_internal"))?
        .map(Json)
        .map_err(catalog_error)
}

async fn info(
    State(app): State<App>,
    body: Result<Json<InfoRequest>, JsonRejection>,
) -> Result<Json<catalog::Info>, Error> {
    let body = input(body)?;
    schema(body.schema_version)?;
    // An absolute manual selection does not depend on a configured catalog root.
    let root = if std::path::Path::new(&body.file).is_absolute() {
        PathBuf::new()
    } else {
        root(&app)?
    };
    tokio::task::spawn_blocking(move || catalog::info(&root, &body.file))
        .await
        .map_err(|_| Error::new(StatusCode::INTERNAL_SERVER_ERROR, "lora_catalog_internal"))?
        .map(Json)
        .map_err(catalog_error)
}
