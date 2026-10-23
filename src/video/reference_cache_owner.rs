//! Session ownership survives one-job workers. No tensor/GPU allocation in the server.
use super::reference_cache::{Cache, Ticket};
use crate::h3_protocol::{
    MAX_PAYLOAD,
    cache::{Command, Descriptor, Reply},
};
use std::sync::{Arc, Mutex};
pub const DEFAULT_LIMIT: u64 = 256 * 1024 * 1024;
pub struct Encoding {
    pub descriptor: Descriptor,
    pub bytes: Vec<u8>,
}
pub type Owner = Arc<Mutex<Cache<Encoding>>>;
pub fn new() -> Owner {
    Arc::new(Mutex::new(Cache::new(DEFAULT_LIMIT)))
}
pub struct Exchange {
    owner: Owner,
    serial: u64,
    pending: Option<(u64, Ticket)>,
    hit: Option<Arc<Encoding>>,
    upload: Option<Encoding>,
}
impl Exchange {
    pub fn new(owner: Owner) -> Self {
        Self {
            owner,
            serial: 0,
            pending: None,
            hit: None,
            upload: None,
        }
    }
    fn abort(&mut self) {
        if let Some((_, ticket)) = self.pending.take()
            && let Ok(mut c) = self.owner.lock()
        {
            c.abort(&ticket);
        }
        self.hit = None;
        self.upload = None;
    }
    fn chunk(&self, offset: usize) -> Result<(Reply, Vec<u8>), &'static str> {
        let e = self.hit.as_ref().ok_or("h3_protocol_error")?;
        if offset >= e.bytes.len() {
            return Err("h3_protocol_error");
        }
        Ok((
            Reply::Chunk {
                descriptor: e.descriptor.clone(),
                offset,
            },
            e.bytes[offset..e.bytes.len().min(offset.saturating_add(MAX_PAYLOAD))].to_vec(),
        ))
    }
    pub fn handle(
        &mut self,
        command: Command,
        payload: &[u8],
    ) -> Result<(Reply, Vec<u8>), &'static str> {
        if !command.valid_payload(payload.len()) {
            return Err("h3_protocol_error");
        }
        match command {
            Command::Lookup { key } => {
                self.abort();
                let mut c = self
                    .owner
                    .lock()
                    .map_err(|_| "reference_cache_unavailable")?;
                self.hit = c.get(&key);
                if self.hit.is_none() {
                    self.serial = self.serial.wrapping_add(1);
                    let ticket = c.begin(key).map(|t| {
                        self.pending = Some((self.serial, t));
                        self.serial
                    });
                    return Ok((Reply::Miss { ticket }, vec![]));
                }
                drop(c);
                self.chunk(0)
            }
            Command::Fetch { offset } => self.chunk(offset),
            Command::Store {
                ticket,
                descriptor,
                offset,
            } => {
                if self.pending.as_ref().is_none_or(|(n, _)| *n != ticket) {
                    return Err("h3_protocol_error");
                }
                let total = descriptor.bytes().ok_or("h3_protocol_error")?;
                let status = self
                    .owner
                    .lock()
                    .map_err(|_| "reference_cache_unavailable")?
                    .status();
                if status.mode == "off" || total as u64 > status.limit_bytes {
                    self.abort();
                    return Ok((
                        Reply::Stored {
                            accepted: false,
                            complete: true,
                        },
                        vec![],
                    ));
                }
                if offset == 0 && self.upload.is_none() {
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(total)
                        .map_err(|_| "reference_cache_allocation_failed")?;
                    self.upload = Some(Encoding {
                        descriptor: descriptor.clone(),
                        bytes,
                    });
                }
                let e = self.upload.as_mut().ok_or("h3_protocol_error")?;
                if e.descriptor != descriptor || e.bytes.len() != offset {
                    return Err("h3_protocol_error");
                }
                e.bytes.extend_from_slice(payload);
                let complete = e.bytes.len() == total;
                let accepted = if complete {
                    let e = self.upload.take().unwrap();
                    let (_, ticket) = self.pending.take().unwrap();
                    self.owner
                        .lock()
                        .map_err(|_| "reference_cache_unavailable")?
                        .finish(ticket, Arc::new(e), total as u64)
                } else {
                    true
                };
                Ok((Reply::Stored { accepted, complete }, vec![]))
            }
        }
    }
}
impl Drop for Exchange {
    fn drop(&mut self) {
        self.abort();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{h3_protocol::cache::Dtype, video::reference_cache::Control};
    fn key(n: u8) -> String {
        format!("{n:064x}")
    }
    fn store(exchange: &mut Exchange, n: u8, values: &[u8]) -> bool {
        let (
            Reply::Miss {
                ticket: Some(ticket),
            },
            _,
        ) = exchange
            .handle(Command::Lookup { key: key(n) }, &[])
            .unwrap()
        else {
            panic!("expected miss");
        };
        let descriptor = Descriptor {
            shape: vec![(values.len() / 2) as i64],
            dtype: Dtype::F16,
            strides: None,
        };
        let mut final_value = false;
        for (i, chunk) in values.chunks(MAX_PAYLOAD).enumerate() {
            let (Reply::Stored { accepted, complete }, body) = exchange
                .handle(
                    Command::Store {
                        ticket,
                        descriptor: descriptor.clone(),
                        offset: i * MAX_PAYLOAD,
                    },
                    chunk,
                )
                .unwrap()
            else {
                panic!("expected store");
            };
            assert!(body.is_empty());
            if complete {
                final_value = accepted;
                break;
            }
        }
        final_value
    }
    #[test]
    fn session_encodings_survive_worker_exchange_and_clear_preserves_active_reader() {
        let owner = new();
        let value = vec![0x5a; MAX_PAYLOAD + 6];
        assert!(store(&mut Exchange::new(owner.clone()), 1, &value));
        let status = owner.status().unwrap();
        assert_eq!(
            (status.entries, status.used_bytes, status.busy),
            (1, value.len() as u64, 0)
        );
        let mut next_worker = Exchange::new(owner.clone());
        let (reply, first) = next_worker
            .handle(Command::Lookup { key: key(1) }, &[])
            .unwrap();
        assert!(matches!(reply, Reply::Chunk { offset: 0, .. }));
        assert_eq!(first, value[..MAX_PAYLOAD]);
        let cleared = owner.clear().unwrap();
        assert_eq!((cleared.entries, cleared.used_bytes), (0, 0));
        let (_, tail) = next_worker
            .handle(
                Command::Fetch {
                    offset: MAX_PAYLOAD,
                },
                &[],
            )
            .unwrap();
        assert_eq!(tail, value[MAX_PAYLOAD..]);
        assert!(matches!(
            Exchange::new(owner)
                .handle(Command::Lookup { key: key(1) }, &[])
                .unwrap()
                .0,
            Reply::Miss { .. }
        ));
    }
    #[test]
    fn clear_during_encode_and_dropped_worker_never_repopulate_or_leave_busy_ticket() {
        let owner = new();
        let mut worker = Exchange::new(owner.clone());
        let (
            Reply::Miss {
                ticket: Some(ticket),
            },
            _,
        ) = worker.handle(Command::Lookup { key: key(1) }, &[]).unwrap()
        else {
            panic!()
        };
        assert_eq!(owner.status().unwrap().busy, 1);
        owner.clear().unwrap();
        assert!(matches!(
            worker
                .handle(
                    Command::Store {
                        ticket,
                        descriptor: Descriptor {
                            shape: vec![1],
                            dtype: Dtype::F16,
                            strides: None
                        },
                        offset: 0
                    },
                    &[1, 2]
                )
                .unwrap()
                .0,
            Reply::Stored {
                accepted: false,
                complete: true
            }
        ));
        assert_eq!(owner.status().unwrap().entries, 0);
        worker.handle(Command::Lookup { key: key(2) }, &[]).unwrap();
        assert_eq!(owner.status().unwrap().busy, 1);
        drop(worker);
        assert_eq!(owner.status().unwrap().busy, 0);
    }
    #[test]
    fn budget_lru_disable_and_malformed_offsets() {
        let owner = new();
        owner.configure(6, true).unwrap();
        let mut worker = Exchange::new(owner.clone());
        assert!(store(&mut worker, 1, &[1, 2, 3, 4]));
        assert!(store(&mut worker, 2, &[5, 6, 7, 8]));
        assert_eq!(owner.status().unwrap().used_bytes, 4);
        assert!(matches!(
            worker
                .handle(Command::Lookup { key: key(1) }, &[])
                .unwrap()
                .0,
            Reply::Miss { .. }
        ));
        assert!(!store(&mut worker, 3, &[1; 8]));
        assert_eq!(owner.status().unwrap().used_bytes, 4);
        owner.configure(0, false).unwrap();
        assert_eq!(owner.status().unwrap().used_bytes, 0);
        assert!(matches!(
            worker
                .handle(Command::Lookup { key: key(2) }, &[])
                .unwrap()
                .0,
            Reply::Miss { ticket: None }
        ));
        assert!(worker.handle(Command::Fetch { offset: 0 }, &[]).is_err());
        assert!(
            worker
                .handle(
                    Command::Lookup {
                        key: "invalid".into()
                    },
                    &[]
                )
                .is_err()
        );
    }
}
