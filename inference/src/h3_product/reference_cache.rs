//! Actual VAE encode seam. Retained CPU bytes belong to the server session.
use super::{
    protocol::{
        self as p, Event,
        cache::{Command, Descriptor, Dtype, Reply},
    },
    wire::Emit,
};
use crate::native::Error;
use sha2::{Digest, Sha256};
use std::{io::Read, os::unix::fs::MetadataExt, path::Path};
use tch::{Device, Kind, Tensor};
fn error() -> Error {
    Error::Native("Reference encoding cache transport failed".into())
}
fn dtype(kind: Kind) -> Option<Dtype> {
    match kind {
        Kind::Half => Some(Dtype::F16),
        Kind::BFloat16 => Some(Dtype::Bf16),
        Kind::Float => Some(Dtype::F32),
        Kind::Double => Some(Dtype::F64),
        _ => None,
    }
}
fn kind(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::F16 => Kind::Half,
        Dtype::Bf16 => Kind::BFloat16,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
    }
}
fn bytes(tensor: &Tensor) -> Result<Vec<u8>, Error> {
    let raw = tensor
        .f_to_device(Device::Cpu)?
        .f_contiguous()?
        .f_view_dtype(Kind::Uint8)?;
    let mut bytes = vec![0; raw.numel()];
    let n = bytes.len();
    raw.f_copy_data_u8(&mut bytes, n)?;
    Ok(bytes)
}
fn identity(path: &Path) -> Result<Vec<u8>, Error> {
    let m = path.metadata().map_err(|_| error())?;
    serde_json::to_vec(&(
        path,
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
    .map_err(|_| error())
}
fn rpc(
    id: &str,
    command: Command,
    payload: &[u8],
    emit: &mut Emit<'_>,
    input: &mut dyn Read,
) -> Result<(Reply, Vec<u8>), Error> {
    emit(
        &Event::ReferenceCache {
            id: id.into(),
            command,
        },
        payload,
    )
    .map_err(|_| error())?;
    fn part(input: &mut dyn Read, limit: usize) -> Result<Vec<u8>, Error> {
        let mut n = [0; 4];
        input.read_exact(&mut n).map_err(|_| error())?;
        let n = u32::from_be_bytes(n) as usize;
        if n > limit {
            return Err(error());
        }
        let mut b = vec![0; n];
        input.read_exact(&mut b).map_err(|_| error())?;
        Ok(b)
    }
    let header = part(input, p::MAX_HEADER)?;
    Ok((
        serde_json::from_slice(&header).map_err(|_| error())?,
        part(input, p::MAX_PAYLOAD)?,
    ))
}
fn lookup(
    id: &str,
    key: String,
    emit: &mut Emit<'_>,
    input: &mut dyn Read,
) -> Result<(Option<Tensor>, Option<u64>), Error> {
    let (reply, chunk) = rpc(id, Command::Lookup { key }, &[], emit, input)?;
    match reply {
        Reply::Miss { ticket } if chunk.is_empty() => Ok((None, ticket)),
        Reply::Chunk {
            descriptor,
            offset: 0,
        } => {
            let total = descriptor.bytes().ok_or_else(error)?;
            if chunk.is_empty() || chunk.len() > total {
                return Err(error());
            }
            let mut data = chunk;
            data.try_reserve_exact(total - data.len())
                .map_err(|_| error())?;
            while data.len() < total {
                let offset = data.len();
                let (reply, chunk) = rpc(id, Command::Fetch { offset }, &[], emit, input)?;
                if !matches!(reply, Reply::Chunk { descriptor: ref d, offset: o } if d == &descriptor && o == offset)
                    || chunk.is_empty()
                    || offset
                        .checked_add(chunk.len())
                        .is_none_or(|end| end > total)
                {
                    return Err(error());
                }
                data.extend_from_slice(&chunk);
            }
            let contiguous = Tensor::f_from_slice(&data)?
                .f_view_dtype(kind(descriptor.dtype))?
                .f_view(descriptor.shape.as_slice())?;
            let tensor = if let Some(strides) = &descriptor.strides {
                let mut value = Tensor::f_empty_strided(
                    descriptor.shape.as_slice(),
                    strides.as_slice(),
                    (kind(descriptor.dtype), Device::Cpu),
                )?;
                value.f_copy_(&contiguous)?;
                value
            } else {
                contiguous
            };
            Ok((Some(tensor), None))
        }
        _ => Err(error()),
    }
}
pub struct Encoder<'a> {
    pub weights: &'a Path,
    pub library: &'a Path,
    pub policy: &'a [u8],
}
pub fn encode(
    id: &str,
    encoder: Encoder<'_>,
    source: &Tensor,
    emit: &mut Emit<'_>,
    input: &mut dyn Read,
    encode: impl FnOnce() -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    // Cache identity is independent of asset aliases. Metadata identifies the selected
    // open model/library version without hashing multi-GB weights on each request.
    let mut hash = Sha256::new();
    for part in [
        b"h3-reference-vae-v1".to_vec(),
        identity(encoder.weights)?,
        identity(encoder.library)?,
        encoder.policy.to_vec(),
        serde_json::to_vec(&(source.size(), format!("{:?}", source.kind())))
            .map_err(|_| error())?,
        bytes(source)?,
    ] {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    let (hit, ticket) = lookup(id, format!("{:x}", hash.finalize()), emit, input)?;
    if let Some(value) = hit {
        return Ok(value);
    }
    let value = encode()?.f_to_device(Device::Cpu)?;
    if let (Some(ticket), Some(dtype)) = (ticket, dtype(value.kind())) {
        let descriptor = Descriptor {
            shape: value.size(),
            dtype,
            strides: Some(value.stride()),
        };
        // Non-dense views still generate normally, without retaining a cache entry.
        if descriptor.bytes().is_none() {
            return Ok(value);
        }
        let raw = bytes(&value)?;
        for (index, chunk) in raw.chunks(p::MAX_PAYLOAD).enumerate() {
            let offset = index * p::MAX_PAYLOAD;
            let (reply, payload) = rpc(
                id,
                Command::Store {
                    ticket,
                    descriptor: descriptor.clone(),
                    offset,
                },
                chunk,
                emit,
                input,
            )?;
            if !payload.is_empty() {
                return Err(error());
            }
            match reply {
                Reply::Stored {
                    accepted: false,
                    complete: true,
                } => break,
                Reply::Stored {
                    accepted: true,
                    complete,
                } if complete == (offset + chunk.len() == raw.len()) => {}
                _ => return Err(error()),
            }
        }
    }
    Ok(value)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, io::Cursor};
    fn frame(reply: Reply, body: &[u8]) -> Vec<u8> {
        let header = serde_json::to_vec(&reply).unwrap();
        [
            (header.len() as u32).to_be_bytes().as_slice(),
            &header,
            (body.len() as u32).to_be_bytes().as_slice(),
            body,
        ]
        .concat()
    }
    #[test]
    fn actual_tensor_cache_hit_is_byte_exact_and_skips_encode_for_all_native_vae_dtypes() {
        let file = std::env::current_exe().unwrap(); // own code artifact metadata only
        for dtype in [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::F64] {
            let source = Tensor::f_from_slice(&[0.125f32, -0.25, 0.5, 1.])
                .unwrap()
                .view([1, 2, 2]);
            let expected = source.to_kind(kind(dtype)).transpose(1, 2);
            let raw = bytes(&expected).unwrap();
            let calls = Cell::new(0);
            let mut events = vec![];
            let mut emit = |e: &Event, b: &[u8]| {
                events.push((e.clone(), b.to_vec()));
                Ok(())
            };
            let replies = [
                frame(Reply::Miss { ticket: Some(7) }, &[]),
                frame(
                    Reply::Stored {
                        accepted: true,
                        complete: true,
                    },
                    &[],
                ),
            ]
            .concat();
            let first = encode(
                "own",
                Encoder {
                    weights: &file,
                    library: &file,
                    policy: b"own-policy",
                },
                &source,
                &mut emit,
                &mut Cursor::new(replies),
                || {
                    calls.set(calls.get() + 1);
                    Ok(expected.shallow_clone())
                },
            )
            .unwrap();
            assert_eq!(calls.get(), 1);
            assert_eq!(bytes(&first).unwrap(), raw);
            let Event::ReferenceCache {
                command: Command::Lookup { key: first_key },
                ..
            } = &events[0].0
            else {
                panic!()
            };
            assert_eq!(events[1].1, raw);
            let descriptor = Descriptor {
                shape: expected.size(),
                dtype,
                strides: Some(expected.stride()),
            };
            let mut hit_events = vec![];
            let mut hit_emit = |e: &Event, _: &[u8]| {
                hit_events.push(e.clone());
                Ok(())
            };
            let hit = encode(
                "own",
                Encoder {
                    weights: &file,
                    library: &file,
                    policy: b"own-policy",
                },
                &source,
                &mut hit_emit,
                &mut Cursor::new(frame(
                    Reply::Chunk {
                        descriptor,
                        offset: 0,
                    },
                    &raw,
                )),
                || panic!("cache hit encoded again"),
            )
            .unwrap();
            assert_eq!(hit.kind(), expected.kind());
            assert_eq!(hit.stride(), expected.stride());
            assert_eq!(hit.size(), expected.size());
            assert_eq!(bytes(&hit).unwrap(), raw);
            assert!(
                matches!(&hit_events[0],Event::ReferenceCache { command:Command::Lookup { key },.. } if key==first_key)
            );
            let mut changed_key = None;
            let mut miss_emit = |e: &Event, _: &[u8]| {
                if let Event::ReferenceCache {
                    command: Command::Lookup { key },
                    ..
                } = e
                {
                    changed_key = Some(key.clone());
                }
                Ok(())
            };
            let _ = encode(
                "own",
                Encoder {
                    weights: &file,
                    library: &file,
                    policy: b"changed-policy",
                },
                &source,
                &mut miss_emit,
                &mut Cursor::new(frame(Reply::Miss { ticket: None }, &[])),
                || Ok(expected.shallow_clone()),
            )
            .unwrap();
            assert_ne!(changed_key.as_ref(), Some(first_key));
        }
    }
}
