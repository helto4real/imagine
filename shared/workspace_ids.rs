//! Workspace keys and file IDs, shared by the server and the MCP proxy.
use sha2::{Digest, Sha256};

const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
pub const KEY_PREFIX: &str = "imk-";
const KEY_CHARS: usize = 26;

fn base32(bytes: &[u8], chars: usize) -> String {
    let mut out = String::with_capacity(chars);
    let (mut buffer, mut bits) = (0u32, 0u32);
    for byte in bytes {
        buffer = (buffer << 8) | u32::from(*byte);
        bits += 8;
        while bits >= 5 && out.len() < chars {
            bits -= 5;
            out.push(ALPHABET[((buffer >> bits) & 31) as usize] as char);
        }
    }
    out
}
/// A new workspace key: `imk-` and 26 base32 characters (130 random bits).
pub fn new_key(random: &[u8; 17]) -> String {
    format!("{KEY_PREFIX}{}", base32(random, KEY_CHARS))
}
/// Whether `key` has the workspace key shape; says nothing about existence.
pub fn key_shaped(key: &str) -> bool {
    key.strip_prefix(KEY_PREFIX)
        .is_some_and(|rest| rest.len() == KEY_CHARS && rest.bytes().all(|b| ALPHABET.contains(&b)))
}
/// The SHA-256 a workspace stores instead of its key.
pub fn key_hash(key: &str) -> String {
    format!("{:x}", Sha256::digest(key.as_bytes()))
}
/// The stable ID of a file at `relative` (slash-separated, inside the
/// workspace folder): the same file keeps its ID across restarts.
pub fn file_id(workspace: &str, relative: &str) -> String {
    let digest = Sha256::digest(format!("{workspace}\0{relative}").as_bytes());
    format!("f{}", base32(&digest, 20))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_and_file_ids_have_fixed_shapes_and_are_stable() {
        let key = new_key(&[0xa5; 17]);
        assert!(key_shaped(&key), "{key}");
        assert_eq!(key.len(), 4 + 26);
        assert!(!key_shaped("imk-short"));
        assert!(!key_shaped(&key.to_uppercase()));
        assert_eq!(key_hash(&key).len(), 64);
        let id = file_id("ws_own", "images/0001.png");
        assert_eq!(id, file_id("ws_own", "images/0001.png"));
        assert_ne!(id, file_id("ws_other", "images/0001.png"));
        assert_eq!(id.len(), 21);
    }
}
