//! Original independent sampler noise and NumPy SeedSequence integer mixing.
use crate::native::{Error, Generator, Kitchen};
use tch::{Device, Tensor};

pub struct DefaultNoise {
    // The original lambda captures x itself and reads its metadata per draw.
    // Metadata caching would both free its storage early and miss resize/data.
    source: Tensor,
    generator: Option<Generator>,
}
impl DefaultNoise {
    pub fn new(native: &Kitchen, x: &Tensor, seed: Option<i128>) -> Result<Self, Error> {
        let generator = if let Some(seed) = seed {
            let seed = if x.device() == Device::Cpu {
                seed.checked_add(1)
            } else {
                Some(seed)
            };
            let seed = seed
                .filter(|s| *s >= i64::MIN as i128 && *s <= u64::MAX as i128)
                .ok_or_else(|| Error::ReferenceException {
                    class: "ValueError",
                    message: "Overflow when unpacking long long".into(),
                })?;
            Some(native.generator(x.device(), seed as u64)?)
        } else {
            None
        };
        Ok(Self {
            source: x.shallow_clone(),
            generator,
        })
    }
    pub fn sample(&mut self) -> Result<Tensor, Error> {
        crate::native::normal_from_source(&self.source, self.generator.as_mut())
    }
    pub fn state(&self) -> Result<Option<Tensor>, Error> {
        self.generator.as_ref().map(Generator::state).transpose()
    }
}
pub fn words(mut n: u128) -> Vec<u32> {
    let mut words = Vec::new();
    loop {
        words.push(n as u32);
        n >>= 32;
        if n == 0 {
            return words;
        }
    }
}
/// Word arrays encode nonnegative arbitrary-size integers, least significant
/// word first. Spawn components are individually encoded, as NumPy does.
pub fn seed_sequence(
    entropy: &[u32],
    spawn: &[Vec<u32>],
    pool_size: usize,
    count: usize,
) -> Result<Vec<u32>, Error> {
    if pool_size < 4 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "The size of the entropy pool should be at least 4".into(),
        });
    }
    let mut assembled = entropy.to_vec();
    if !spawn.is_empty() && assembled.len() < pool_size {
        assembled.resize(pool_size, 0);
    }
    for component in spawn {
        assembled.extend(component);
    }
    fn hash(mut value: u32, constant: &mut u32) -> u32 {
        value ^= *constant;
        *constant = constant.wrapping_mul(0x931e8875);
        value = value.wrapping_mul(*constant);
        value ^ (value >> 16)
    }
    fn mix(a: u32, b: u32) -> u32 {
        let n = 0xca01f9dd_u32
            .wrapping_mul(a)
            .wrapping_sub(0x4973f715_u32.wrapping_mul(b));
        n ^ (n >> 16)
    }
    let mut constant = 0x43b0d7e5;
    let mut pool = (0..pool_size)
        .map(|i| hash(assembled.get(i).copied().unwrap_or(0), &mut constant))
        .collect::<Vec<_>>();
    for source in 0..pool_size {
        for destination in 0..pool_size {
            if source != destination {
                pool[destination] = mix(pool[destination], hash(pool[source], &mut constant));
            }
        }
    }
    for &word in assembled.iter().skip(pool_size) {
        for value in &mut pool {
            *value = mix(*value, hash(word, &mut constant));
        }
    }
    let mut constant = 0x8b51f9dd_u32;
    Ok((0..count)
        .map(|i| {
            let mut n = pool[i % pool_size] ^ constant;
            constant = constant.wrapping_mul(0x58f38ded);
            n = n.wrapping_mul(constant);
            n ^ (n >> 16)
        })
        .collect())
}
