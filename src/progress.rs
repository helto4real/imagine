//! Numeric telemetry from the hash-pinned engine's pretty_progress output.
//! Preview generation stays disabled. No raw output is retained or forwarded.
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingProgress {
    pub completed: u32,
    pub total: u32,
    pub elapsed_seconds: f64,
    pub remaining_seconds: f64,
    pub updated_at_ms: u64,
}

pub fn parse_step(record: &[u8]) -> Option<(u32, u32, f64)> {
    let text = std::str::from_utf8(record).ok()?.trim();
    let text = text.strip_prefix('|')?;
    let (bar, rest) = text.split_once('|')?;
    if bar.len() != 50 || !bar.bytes().all(|c| matches!(c, b'=' | b'>' | b' ')) {
        return None;
    }
    let (counts, speed) = rest.trim().split_once(" - ")?;
    let (step, total) = counts.split_once('/')?;
    let step: u32 = step.parse().ok()?;
    let total: u32 = total.parse().ok()?;
    let speed = speed.strip_suffix("\u{1b}[K").unwrap_or(speed);
    let seconds = if let Some(value) = speed.strip_suffix("s/it") {
        value.parse::<f64>().ok()?
    } else {
        1.0 / speed.strip_suffix("it/s")?.parse::<f64>().ok()?
    };
    // Reject loading byte counters (MB/s), malformed output and unsupported counts.
    (step > 0 && step <= total && seconds.is_finite() && seconds > 0.)
        .then_some((step, total, seconds))
}

pub async fn read_steps(mut reader: impl AsyncRead + Unpin, mut report: impl FnMut(u32, u32, f64)) {
    let mut chunk = [0; 4096];
    let mut record = Vec::with_capacity(256);
    let mut overflow = false;
    while let Ok(count) = reader.read(&mut chunk).await {
        if count == 0 {
            break;
        }
        for &byte in &chunk[..count] {
            if matches!(byte, b'\r' | b'\n') {
                if !overflow && let Some((step, total, seconds)) = parse_step(&record) {
                    report(step, total, seconds);
                }
                record.clear();
                overflow = false;
            } else if record.len() < 256 {
                record.push(byte);
                // Native intermediate steps end with ESC[K, without a newline.
                if record.ends_with(b"\x1b[K") {
                    if !overflow && let Some((step, total, seconds)) = parse_step(&record) {
                        report(step, total, seconds);
                    }
                    record.clear();
                    overflow = true;
                }
            } else {
                overflow = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn positive_130_step_counts_are_numeric_telemetry() {
        let bar = "=".repeat(50);
        assert_eq!(
            super::parse_step(format!("|{bar}| 130/130 - 0.5s/it").as_bytes()),
            Some((130, 130, 0.5))
        );
        for counts in ["0/130", "131/130", "1/0"] {
            assert!(super::parse_step(format!("|{bar}| {counts} - 0.5s/it").as_bytes()).is_none());
        }
        assert!(super::parse_step(format!("|{bar}| 1/130 - NaNs/it").as_bytes()).is_none());
    }

    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn parses_split_live_records_without_waiting_for_newline() {
        let (mut writer, reader) = tokio::io::duplex(4096);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(read_steps(reader, move |s, t, v| {
            tx.send((s, t, v)).unwrap();
        }));
        let valid = format!("\r  |{}| 1/40 - 0.80s/it\x1b[K", "=".repeat(50));
        writer
            .write_all(b"unrelated private content\n")
            .await
            .unwrap();
        writer.write_all(&vec![b'x'; 1024]).await.unwrap();
        writer.write_all(b"\r").await.unwrap();
        for chunk in valid.as_bytes().chunks(3) {
            writer.write_all(chunk).await.unwrap();
        }
        // No newline or EOF: an intermediate native step must already be available.
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap(),
            Some((1, 40, 0.8))
        );
        let bytes = format!("\r  |{}| 2/40 - 2.00GB/s\x1b[K\n", "=".repeat(50));
        writer.write_all(bytes.as_bytes()).await.unwrap();
        drop(writer);
        task.await.unwrap();
        assert!(rx.recv().await.is_none());
        let record = |s: &str| format!("  |{}| {s}\x1b[K", "=".repeat(50));
        assert_eq!(
            parse_step(record("2/40 - 2.00it/s").as_bytes()),
            Some((2, 40, 0.5))
        );
        for invalid in [
            "41/40 - 1.00s/it",
            "1/0 - 1.00s/it",
            "1/40 - 0.00it/s",
            "1/40 - NaNs/it",
            "1/40 - 20.0MB/s",
        ] {
            assert!(parse_step(record(invalid).as_bytes()).is_none());
        }
    }
}
