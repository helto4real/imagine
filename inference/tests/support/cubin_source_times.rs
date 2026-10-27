//! `ptxas -lineinfo` writes each source file's modification time into a cubin's
//! DWARF line table. Reinstalling Kitchen rewrites its unchanged Triton sources with
//! new times, so cubins are compared with the times the original capture compiled
//! with: only those fields are set, every other byte stays exact.
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Recorded capture times: (path below site-packages, seconds).
pub type Times = Vec<(String, u64)>;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/triton-capture-source-times.json"
    ))
    .unwrap()
}
/// Times the stage-1 capture of one source group compiled with.
pub fn stage1(group: &str) -> Times {
    times(&fixture()["captures"][group])
}
/// Times from a capture's `compiler_source_files` records, or `{file, mtime}` rows.
pub fn times(rows: &Value) -> Times {
    rows.as_array()
        .unwrap()
        .iter()
        .map(|row| match row.get("mtime_ns") {
            Some(ns) => (
                row["reference_path"].as_str().unwrap().into(),
                ns.as_u64().unwrap() / 1_000_000_000,
            ),
            None => (
                row["file"].as_str().unwrap().into(),
                row["mtime"].as_u64().unwrap(),
            ),
        })
        .collect()
}
/// SHA-256 of `cubin` as compiled from sources with the given times.
pub fn digest(cubin: &[u8], times: &Times) -> String {
    Sha256::digest(with_source_times(cubin, times))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn le(bytes: &[u8], at: usize, width: usize) -> usize {
    bytes[at..at + width]
        .iter()
        .rev()
        .fold(0, |value, &b| value << 8 | usize::from(b))
}
fn cstr(bytes: &[u8], at: usize) -> &[u8] {
    let end = bytes[at..].iter().position(|&b| b == 0).unwrap();
    &bytes[at..at + end]
}
fn uleb_len(bytes: &[u8], at: usize) -> usize {
    bytes[at..].iter().position(|&b| b & 0x80 == 0).unwrap() + 1
}
fn uleb(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}
/// `cubin` (ELF64, DWARF 2-4 line tables) with the time of each file entry named
/// like a recorded source replaced by the recorded time.
pub fn with_source_times(cubin: &[u8], times: &Times) -> Vec<u8> {
    let mut out = cubin.to_vec();
    let (shoff, shentsize, shnum, shstrndx) = (
        le(cubin, 0x28, 8),
        le(cubin, 0x3a, 2),
        le(cubin, 0x3c, 2),
        le(cubin, 0x3e, 2),
    );
    let names = le(cubin, shoff + shstrndx * shentsize + 0x18, 8);
    for index in 0..shnum {
        let header = shoff + index * shentsize;
        if cstr(cubin, names + le(cubin, header, 4)) != b".debug_line" {
            continue;
        }
        let (start, size) = (le(cubin, header + 0x18, 8), le(cubin, header + 0x20, 8));
        let mut unit = start;
        while unit < start + size {
            let version = le(cubin, unit + 4, 2);
            // minimum_instruction_length, [maximum_operations_per_instruction],
            // default_is_stmt, line_base, line_range, then opcode_base.
            let mut at = unit + 10 + 4 + usize::from(version >= 4);
            at += cubin[at] as usize;
            while cubin[at] != 0 {
                at += cstr(cubin, at).len() + 1;
            }
            at += 1;
            while cubin[at] != 0 {
                let name = cstr(cubin, at);
                at += name.len() + 1;
                at += uleb_len(cubin, at);
                let width = uleb_len(cubin, at);
                if let Some((_, time)) = times
                    .iter()
                    .find(|(file, _)| file.rsplit('/').next().unwrap().as_bytes() == name)
                {
                    let encoded = uleb(*time);
                    assert_eq!(encoded.len(), width, "source time field width");
                    out[at..at + width].copy_from_slice(&encoded);
                }
                at += width;
                at += uleb_len(cubin, at);
            }
            unit += 4 + le(cubin, unit, 4);
        }
    }
    out
}
