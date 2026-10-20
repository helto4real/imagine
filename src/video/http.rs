//! Session-owned media delivery with bounded streaming and single byte ranges.
use crate::{scheduler::Fault, storage::Store};
use axum::{
    body::Body,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use std::io::SeekFrom;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
fn range(value: &str, size: u64) -> Option<(u64, u64)> {
    let (left, right) = value.strip_prefix("bytes=")?.split_once('-')?;
    if size == 0 || right.contains(',') {
        return None;
    }
    if left.is_empty() {
        let suffix = right.parse::<u64>().ok()?;
        if suffix == 0 {
            return None;
        }
        return Some((size.saturating_sub(suffix), size - 1));
    }
    let start = left.parse::<u64>().ok()?;
    if start >= size {
        return None;
    }
    let end = if right.is_empty() {
        size - 1
    } else {
        right.parse::<u64>().ok()?.min(size - 1)
    };
    (end >= start).then_some((start, end))
}
pub async fn download(store: &Store, id: &str, headers: &HeaderMap) -> Result<Response, Fault> {
    let info = store
        .asset(id, "local")
        .await?
        .ok_or(Fault::new(404, "asset_not_found"))?;
    let size = info["bytes"]
        .as_u64()
        .ok_or(Fault::new(500, "storage_error"))?;
    let etag = format!("\"{}\"", info["sha256"].as_str().unwrap_or_default());
    let requested = headers.get(header::RANGE).filter(|_| {
        headers
            .get(header::IF_RANGE)
            .is_none_or(|v| v.as_bytes() == etag.as_bytes())
    });
    let selected = if let Some(value) = requested {
        match value.to_str().ok().and_then(|v| range(v, size)) {
            Some(r) => Some(r),
            None => {
                return Ok((
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{size}"))],
                )
                    .into_response());
            }
        }
    } else {
        None
    };
    let (start, end) = selected.unwrap_or((0, size.saturating_sub(1)));
    let count = if size == 0 { 0 } else { end - start + 1 };
    // Database ownership check above is the only path source; no supplied file paths.
    let mut file = tokio::fs::File::open(store.media.join(id))
        .await
        .map_err(|_| Fault::new(500, "storage_error"))?;
    file.seek(SeekFrom::Start(start))
        .await
        .map_err(|_| Fault::new(500, "storage_error"))?;
    let stream = async_stream::stream! {
        let mut remaining=count;
        while remaining>0 {
            let mut bytes=vec![0u8;remaining.min(64*1024) as usize];
            match file.read_exact(&mut bytes).await {
                Ok(_)=>{remaining-=bytes.len() as u64;yield Ok::<_,std::io::Error>(axum::body::Bytes::from(bytes));}
                Err(error)=>{yield Err(error);break;}
            }
        }
    };
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = if selected.is_some() {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    let h = response.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        info["mime_type"].as_str().unwrap().parse().unwrap(),
    );
    h.insert(header::CONTENT_DISPOSITION, "attachment".parse().unwrap());
    h.insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
    h.insert(header::CONTENT_LENGTH, count.into());
    h.insert(header::ETAG, etag.parse().unwrap());
    if selected.is_some() {
        h.insert(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{size}").parse().unwrap(),
        );
    }
    Ok(response)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_ranges_are_exact_and_invalid_requests_do_not_overflow() {
        assert_eq!(range("bytes=2-4", 10), Some((2, 4)));
        assert_eq!(range("bytes=8-", 10), Some((8, 9)));
        assert_eq!(range("bytes=-3", 10), Some((7, 9)));
        assert_eq!(range("bytes=-30", 10), Some((0, 9)));
        for v in [
            "bytes=10-",
            "bytes=5-4",
            "bytes=0-1,3-4",
            "bytes=-0",
            "bytes=18446744073709551616-",
            "bad",
        ] {
            assert_eq!(range(v, 10), None);
        }
    }
}
