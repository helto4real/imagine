//! CPython UTF-8/ASCII decode results used by checkpoint byte strings.
//! Preserve code points, including surrogatepass, rather than replacement text.
use crate::{
    native::Error,
    patching::options::{Options, SourceText},
};
fn failure(encoding: &str, bytes: &[u8], start: usize, end: usize, reason: &str) -> Error {
    let location = if end - start == 1 {
        format!("byte 0x{:02x} in position {start}", bytes[start])
    } else {
        format!("bytes in position {start}-{}", end - 1)
    };
    Error::ReferenceException {
        class: "UnicodeDecodeError",
        message: format!("'{encoding}' codec can't decode {location}: {reason}"),
    }
}
pub fn ascii(bytes: &[u8]) -> Result<Options, Error> {
    let mut points = Vec::with_capacity(bytes.len());
    for (index, byte) in bytes.iter().enumerate() {
        if *byte > 127 {
            return Err(failure(
                "ascii",
                bytes,
                index,
                index + 1,
                "ordinal not in range(128)",
            ));
        }
        points.push(u32::from(*byte));
    }
    SourceText::from_codepoints(points)
}
pub fn utf8(bytes: &[u8], surrogate_pass: bool) -> Result<Options, Error> {
    let mut points = vec![];
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte < 128 {
            points.push(u32::from(byte));
            index += 1;
            continue;
        }
        // surrogatepass only recovers a complete surrogate sequence. ED A0
        // without its last byte retains the strict decoder's original error.
        if surrogate_pass
            && byte == 0xed
            && index + 2 < bytes.len()
            && (0xa0..=0xbf).contains(&bytes[index + 1])
            && (0x80..=0xbf).contains(&bytes[index + 2])
        {
            points.push(
                (u32::from(byte & 15) << 12)
                    | (u32::from(bytes[index + 1] & 63) << 6)
                    | u32::from(bytes[index + 2] & 63),
            );
            index += 3;
            continue;
        }
        let (length, mut point, low, high) = match byte {
            0xc2..=0xdf => (2, u32::from(byte & 31), 0x80, 0xbf),
            0xe0 => (3, u32::from(byte & 15), 0xa0, 0xbf),
            0xe1..=0xec | 0xee..=0xef => (3, u32::from(byte & 15), 0x80, 0xbf),
            0xed => (3, u32::from(byte & 15), 0x80, 0x9f),
            0xf0 => (4, u32::from(byte & 7), 0x90, 0xbf),
            0xf1..=0xf3 => (4, u32::from(byte & 7), 0x80, 0xbf),
            0xf4 => (4, u32::from(byte & 7), 0x80, 0x8f),
            _ => {
                return Err(failure(
                    "utf-8",
                    bytes,
                    index,
                    index + 1,
                    "invalid start byte",
                ));
            }
        };
        for offset in 1..length {
            let Some(next) = bytes.get(index + offset) else {
                return Err(failure(
                    "utf-8",
                    bytes,
                    index,
                    bytes.len(),
                    "unexpected end of data",
                ));
            };
            if !(0x80..=0xbf).contains(next) || (offset == 1 && !(low..=high).contains(next)) {
                return Err(failure(
                    "utf-8",
                    bytes,
                    index,
                    index + offset,
                    "invalid continuation byte",
                ));
            }
            point = (point << 6) | u32::from(next & 63);
        }
        points.push(point);
        index += length;
    }
    SourceText::from_codepoints(points)
}
