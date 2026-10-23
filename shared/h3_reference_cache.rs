//! Private CPU tensor cache exchange. Bytes are chunked by the existing frame bound.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    F16,
    Bf16,
    F32,
    F64,
}
impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub shape: Vec<i64>,
    pub dtype: Dtype,
    #[serde(default)]
    pub strides: Option<Vec<i64>>,
}
impl Descriptor {
    pub fn dense_layout(&self) -> bool {
        let Some(strides) = &self.strides else {
            return true;
        };
        if strides.len() != self.shape.len() || strides.iter().any(|s| *s < 0) {
            return false;
        }
        let mut dimensions = self
            .shape
            .iter()
            .zip(strides)
            .filter(|(d, _)| **d > 1)
            .map(|(&d, &s)| (s, d))
            .collect::<Vec<_>>();
        dimensions.sort_unstable();
        let mut extent = 1_i64;
        for (stride, dimension) in dimensions {
            if stride != extent {
                return false;
            }
            let Some(next) = extent.checked_mul(dimension) else {
                return false;
            };
            extent = next;
        }
        true
    }
    pub fn bytes(&self) -> Option<usize> {
        if !self.dense_layout() {
            return None;
        }
        self.shape.iter().try_fold(self.dtype.size(), |n, &d| {
            n.checked_mul(usize::try_from(d).ok().filter(|d| *d > 0)?)
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Lookup {
        key: String,
    },
    Fetch {
        offset: usize,
    },
    Store {
        ticket: u64,
        descriptor: Descriptor,
        offset: usize,
    },
}
impl Command {
    pub fn valid_payload(&self, bytes: usize) -> bool {
        match self {
            Self::Lookup { key } => super::is_sha256(key) && bytes == 0,
            Self::Fetch { .. } => bytes == 0,
            Self::Store {
                descriptor, offset, ..
            } => {
                bytes > 0
                    && bytes <= super::MAX_PAYLOAD
                    && descriptor
                        .bytes()
                        .is_some_and(|n| offset.checked_add(bytes).is_some_and(|end| end <= n))
            }
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reply {
    Miss {
        ticket: Option<u64>,
    },
    Chunk {
        descriptor: Descriptor,
        offset: usize,
    },
    Stored {
        accepted: bool,
        complete: bool,
    },
}
