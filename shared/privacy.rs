//! Read Helto's unlocked session only when a profile actually needs decryption.
//! No passwords, keys or decrypted prompt text are logged or written by this module.
use aes_gcm::{
    Aes256Gcm, KeyInit, Nonce,
    aead::{Aead, Payload},
};
use anyhow::{Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::Value;
use std::{io::Read, path::PathBuf};
use zeroize::Zeroizing;
fn session_path() -> PathBuf {
    if let Ok(v) = std::env::var("HELTO_PRIVACY_SESSION_DIR")
        && !v.trim().is_empty()
    {
        return PathBuf::from(v).join("privacy_session.json");
    }
    if let Ok(v) = std::env::var("XDG_RUNTIME_DIR")
        && !v.trim().is_empty()
    {
        return PathBuf::from(v).join("helto/privacy_session.json");
    }
    std::env::temp_dir().join(format!("helto-privacy-{}/privacy_session.json", unsafe {
        libc::getuid()
    }))
}
pub fn decrypt(text: &str) -> Result<String> {
    let Ok(envelope) = serde_json::from_str::<Value>(text) else {
        return Ok(text.into());
    };
    if envelope["encrypted"] != true || envelope["algorithm"] != "AES-256-GCM" {
        return Ok(text.into());
    }
    if envelope["schema"] == "helto.aio-image-generate" {
        bail!(
            "Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore."
        )
    }
    if envelope["schema"] != "helto.aio-image-generate.v2" {
        return Ok(text.into());
    }
    let mut bytes = Zeroizing::new(Vec::new());
    std::fs::File::open(session_path())
        .and_then(|f| f.take(1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|_| {
            anyhow!(
                "PRIVACY_KEYSTORE_LOCKED: Privacy keystore is locked. Authenticate to unlock it."
            )
        })?;
    if bytes.len() > 1024 * 1024 {
        bail!("PRIVACY_KEYSTORE_LOCKED: invalid privacy session")
    }
    decrypt_with_session(&envelope, &bytes)
}
fn decrypt_with_session(envelope: &Value, bytes: &[u8]) -> Result<String> {
    let session: Value = serde_json::from_slice(bytes)
        .map_err(|_| anyhow!("PRIVACY_KEYSTORE_LOCKED: invalid privacy session"))?;
    let keys = session["keys"]
        .as_array()
        .ok_or_else(|| anyhow!("PRIVACY_KEYSTORE_LOCKED: invalid privacy session"))?;
    if session["token"].as_str().is_none_or(|v| v.is_empty())
        || !keys.iter().any(|k| k["keyId"] == session["primaryKeyId"])
    {
        bail!("PRIVACY_KEYSTORE_LOCKED: invalid privacy session")
    }
    let key_id = envelope["keyId"]
        .as_str()
        .ok_or_else(|| anyhow!("PRIVACY_PAYLOAD_INVALID"))?;
    let entry=keys.iter().find(|k|k["keyId"]==key_id).ok_or_else(||anyhow!("PRIVACY_KEY_MISMATCH: Encrypted state payload was created with a different local privacy key."))?;
    let key = Zeroizing::new(
        URL_SAFE_NO_PAD
            .decode(entry["key"].as_str().unwrap_or("").trim_end_matches('='))
            .map_err(|_| anyhow!("PRIVACY_KEY_INVALID"))?,
    );
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| anyhow!("PRIVACY_KEY_INVALID"))?;
    let nonce = URL_SAFE_NO_PAD.decode(
        envelope["nonce"]
            .as_str()
            .unwrap_or("")
            .trim_end_matches('='),
    );
    let ciphertext = URL_SAFE_NO_PAD.decode(
        envelope["ciphertext"]
            .as_str()
            .unwrap_or("")
            .trim_end_matches('='),
    );
    let (Ok(nonce), Ok(ciphertext)) = (nonce, ciphertext) else {
        bail!("PRIVACY_PAYLOAD_INVALID")
    };
    if nonce.len() != 12 {
        bail!("PRIVACY_PAYLOAD_INVALID")
    }
    let aad = format!("helto.aio-image-generate.v2|1|AES-256-GCM|{key_id}");
    let plaintext = Zeroizing::new(
        cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &ciphertext,
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| anyhow!("PRIVACY_DECRYPT_FAILED: Could not decrypt state payload"))?,
    );
    let value: Value =
        serde_json::from_slice(&plaintext).map_err(|_| anyhow!("PRIVACY_PAYLOAD_INVALID"))?;
    if !value.is_object() {
        bail!("PRIVACY_PAYLOAD_INVALID")
    };
    Ok(value["value"].as_str().unwrap_or("").into())
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn synthetic_envelope_and_wrong_key() {
        let key = [17u8; 32];
        let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
        let nonce = [23u8; 12];
        let aad = b"helto.aio-image-generate.v2|1|AES-256-GCM|test";
        let encrypted = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: br#"{"value":"synthetic prompt"}"#,
                    aad,
                },
            )
            .unwrap();
        let mut e = json!({"keyId":"test","nonce":URL_SAFE_NO_PAD.encode(nonce),"ciphertext":URL_SAFE_NO_PAD.encode(encrypted)});
        let session=serde_json::to_vec(&json!({"token":"synthetic","primaryKeyId":"test","keys":[{"keyId":"test","key":URL_SAFE_NO_PAD.encode(key)}]})).unwrap();
        assert_eq!(
            decrypt_with_session(&e, &session).unwrap(),
            "synthetic prompt"
        );
        e["keyId"] = json!("wrong");
        assert!(decrypt_with_session(&e, &session).is_err());
    }
}
