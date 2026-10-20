//! Frozen CPython 3.12.13 LP64 hash mechanics. Hash state is process-owned and
//! separate from inference RNG. No Python runtime or fixed default seed.
use crate::native::Error;
use std::rc::Rc;

pub fn object_identity_hash<T: ?Sized>(owner: &Rc<T>) -> i64 {
    identity_hash_address(Rc::as_ptr(owner) as *const () as usize)
}
pub fn identity_hash_address(address: usize) -> i64 {
    const { assert!(usize::BITS == 64) };
    finish(address.rotate_right(4) as u64)
}
fn finish(value: u64) -> i64 {
    if value == u64::MAX { -2 } else { value as i64 }
}
const NUMERIC_MODULUS: u128 = (1u128 << 61) - 1;
fn signed_numeric(magnitude: u128, negative: bool) -> i64 {
    let value = magnitude as i64;
    let value = if negative { -value } else { value };
    if value == -1 { -2 } else { value }
}
/// Source int.from_bytes(..., little, signed=True) hash, with arbitrary width.
/// This needs neither a host-sized integer truncation nor a large allocation.
pub fn hash_signed_integer_bytes(bytes: &[u8]) -> i64 {
    let negative = bytes.last().is_some_and(|byte| byte & 128 != 0);
    let mut carry = u16::from(negative);
    let (mut remainder, mut power) = (0u128, 1u128);
    for &byte in bytes {
        let digit = if negative {
            let value = u16::from(!byte) + carry;
            carry = value >> 8;
            value & 255
        } else {
            u16::from(byte)
        };
        remainder = (remainder + u128::from(digit) * power) % NUMERIC_MODULUS;
        power = (power * 256) % NUMERIC_MODULUS;
    }
    signed_numeric(remainder, negative)
}
pub fn hash_integer(value: i128) -> i64 {
    hash_signed_integer_bytes(&value.to_le_bytes())
}
/// NaN uses its owning object's generic identity hash. For a complex value,
/// both NaN components use that complex object's owner, not temporary floats.
pub fn hash_float(value: f64, owner_hash: i64) -> i64 {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 2047 {
        return if fraction != 0 {
            if owner_hash == -1 { -2 } else { owner_hash }
        } else {
            signed_numeric(314159, bits >> 63 != 0)
        };
    }
    let (significand, power) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent - 1023 - 52)
    };
    // 2**61 is congruent to 1 modulo the LP64 source modulus.
    let shift = power.rem_euclid(61) as u32;
    signed_numeric(
        (u128::from(significand) << shift) % NUMERIC_MODULUS,
        bits >> 63 != 0,
    )
}
pub fn hash_complex(real: f64, imaginary: f64, owner_hash: i64) -> i64 {
    let value = hash_float(real, owner_hash)
        .wrapping_add(hash_float(imaginary, owner_hash).wrapping_mul(1_000_003));
    if value == -1 { -2 } else { value }
}
#[derive(Clone)]
pub struct HashState {
    secret: [u8; 24],
}
impl HashState {
    /// One ordinary random source hash secret for the native process. Explicit
    /// captured/seeded contexts use from_secret/from_seed instead of replacing
    /// this shared random context or borrowing the inference RNG.
    pub fn shared_random() -> Result<Self, Error> {
        static STATE: std::sync::Mutex<Option<HashState>> = std::sync::Mutex::new(None);
        let mut state = STATE
            .lock()
            .map_err(|_| Error::Native("process hash state lock poisoned".into()))?;
        if let Some(state) = state.as_ref() {
            return Ok(state.clone());
        }
        let value = Self::initialize(None)?;
        *state = Some(value.clone());
        Ok(value)
    }
    /// The source's explicit PYTHONHASHSEED expansion. Zero disables random
    /// hashing; nonzero seeds initialize all 24 bytes using the original LCG.
    pub fn from_seed(seed: u32) -> Self {
        let mut secret = [0; 24];
        if seed != 0 {
            let mut state = seed;
            for byte in &mut secret {
                state = state.wrapping_mul(214013).wrapping_add(2531011);
                *byte = (state >> 16) as u8;
            }
        }
        Self { secret }
    }
    /// Controlled original-process state, including a captured random state.
    /// Ordinary native initialization uses initialize(None), not a test secret.
    pub fn from_secret(secret: [u8; 24]) -> Self {
        Self { secret }
    }
    pub fn secret(&self) -> &[u8; 24] {
        &self.secret
    }
    /// The execution owner initializes this once and retains the resulting
    /// state. None matches ordinary random hashing; Some matches explicit seed.
    pub fn initialize(seed: Option<u32>) -> Result<Self, Error> {
        if let Some(seed) = seed {
            return Ok(Self::from_seed(seed));
        }
        let mut secret = [0_u8; 24];
        let mut done = 0;
        while done < secret.len() {
            let count = unsafe {
                libc::getrandom(
                    secret[done..].as_mut_ptr().cast(),
                    secret.len() - done,
                    libc::GRND_NONBLOCK,
                )
            };
            if count > 0 {
                done += count as usize;
                continue;
            }
            let cause = std::io::Error::last_os_error();
            if count < 0 && cause.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            if count < 0
                && matches!(
                    cause.raw_os_error(),
                    Some(libc::ENOSYS | libc::EPERM | libc::EAGAIN)
                )
            {
                use std::io::Read;
                std::fs::File::open("/dev/urandom")
                    .and_then(|mut file| file.read_exact(&mut secret[done..]))
                    .map_err(|e| {
                        Error::Native(format!("hash secret initialization failed: {e}"))
                    })?;
                break;
            }
            return Err(Error::Native(format!(
                "hash secret initialization failed: {cause}"
            )));
        }
        Ok(Self { secret })
    }
    pub fn hash_bytes(&self, bytes: &[u8]) -> i64 {
        if bytes.is_empty() {
            return 0;
        }
        let key0 = u64::from_le_bytes(self.secret[..8].try_into().unwrap());
        let key1 = u64::from_le_bytes(self.secret[8..16].try_into().unwrap());
        let mut v = [
            key0 ^ 0x736f6d6570736575,
            key1 ^ 0x646f72616e646f6d,
            key0 ^ 0x6c7967656e657261,
            key1 ^ 0x7465646279746573,
        ];
        let mut words = bytes.chunks_exact(8);
        for word in words.by_ref() {
            let word = u64::from_le_bytes(word.try_into().unwrap());
            v[3] ^= word;
            round(&mut v);
            v[0] ^= word;
        }
        let mut last = (bytes.len() as u64) << 56;
        for (i, byte) in words.remainder().iter().enumerate() {
            last |= u64::from(*byte) << (8 * i);
        }
        v[3] ^= last;
        round(&mut v);
        v[0] ^= last;
        v[2] ^= 0xff;
        for _ in 0..3 {
            round(&mut v);
        }
        finish(v[0] ^ v[1] ^ v[2] ^ v[3])
    }
    /// Unicode hashes use original PEP-393 code-unit bytes, not UTF-8 bytes.
    pub fn hash_text(&self, value: &str) -> i64 {
        self.hash_unicode(&value.chars().map(u32::from).collect::<Vec<_>>())
    }
    /// Original Unicode code points, retaining lone surrogates without lossy
    /// UTF-8 conversion. The Unicode object binding validates the upper bound.
    pub fn hash_unicode(&self, value: &[u32]) -> i64 {
        let maximum = value.iter().copied().max().unwrap_or(0);
        let bytes: Vec<u8> = if maximum <= 0xff {
            value.iter().map(|v| *v as u8).collect()
        } else if maximum <= 0xffff {
            value
                .iter()
                .flat_map(|v| (*v as u16).to_le_bytes())
                .collect()
        } else {
            value.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        self.hash_bytes(&bytes)
    }
    pub fn hash_text_tuple(&self, values: &[&str]) -> i64 {
        hash_tuple(&values.iter().map(|v| self.hash_text(v)).collect::<Vec<_>>())
    }
}
fn round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[2] = v[2].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[1]);
    v[0] = v[0].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].rotate_left(32);
}
pub fn hash_tuple(hashes: &[i64]) -> i64 {
    let mut accumulator = 2870177450012600261_u64;
    for &hash in hashes {
        accumulator = accumulator.wrapping_add((hash as u64).wrapping_mul(14029467366897019727));
        accumulator = accumulator
            .rotate_left(31)
            .wrapping_mul(11400714785074694791);
    }
    accumulator = accumulator.wrapping_add((hashes.len() as u64) ^ (2870177450012600261 ^ 3527539));
    if accumulator == u64::MAX {
        1546275796
    } else {
        accumulator as i64
    }
}
