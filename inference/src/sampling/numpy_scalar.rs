//! NumPy 2.3.5 scalar-lane log used by DEIS on the locked initial platform.
//! Float32 uses the original AVX2/FMA P/Q lane algorithm; Float64 and Float16
//! use their original baseline libm loops. These are distinct from ATen.log.
use crate::native::Error;
use tch::{Kind, Tensor};

pub fn log_f32_avx2(mut x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    if x < 0. {
        return f32::from_bits(0xffc0_0000);
    }
    if x == 0. {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        return x;
    }
    let denormal = x < f32::MIN_POSITIVE;
    if denormal {
        x *= f32::from_bits(0x7180_0000);
    }
    let mut exponent = ((x.to_bits() >> 23) as i32 - 126) as f32;
    if denormal {
        exponent -= 100.;
    }
    x = f32::from_bits((x.to_bits() & 0x007f_ffff) | (126 << 23));
    if x <= f32::from_bits(0x3f35_04f3) {
        x += x;
        exponent -= 1.;
    }
    x -= 1.;
    let p = [
        f32::from_bits(0x00000000),
        f32::from_bits(0x3f800000),
        f32::from_bits(0x4007361c),
        f32::from_bits(0x3fbd70a9),
        f32::from_bits(0x3ec30333),
        f32::from_bits(0x3cd42bcd),
    ];
    let q = [
        f32::from_bits(0x3f800000),
        f32::from_bits(0x4027361c),
        f32::from_bits(0x401cfe0d),
        f32::from_bits(0x3f7c8ae4),
        f32::from_bits(0x3e1e5bf3),
        f32::from_bits(0x3bc083df),
    ];
    let mut numerator = p[5].mul_add(x, p[4]);
    for k in (0..4).rev() {
        numerator = numerator.mul_add(x, p[k]);
    }
    let mut denominator = q[5].mul_add(x, q[4]);
    for k in (0..4).rev() {
        denominator = denominator.mul_add(x, q[k]);
    }
    exponent.mul_add(f32::from_bits(0x3f31_7218), numerator / denominator)
}
/// The locked NumPy Float64 loop preserves NaN payload/sign, but its invalid
/// negative-domain log produces positive canonical NaN on this platform.
pub fn log_f64_numpy(x: f64) -> f64 {
    if x < 0. { f64::NAN } else { x.ln() }
}

pub fn log_scalar(value: &Tensor) -> Result<Tensor, Error> {
    Ok(match value.kind() {
        Kind::Float => Tensor::from(log_f32_avx2(value.double_value(&[]) as f32)),
        Kind::Double => Tensor::from(log_f64_numpy(value.double_value(&[]))),
        Kind::Half => {
            Tensor::from(half::f16::from_f32((value.double_value(&[]) as f32).ln()).to_f32())
                .f_to_kind(Kind::Half)?
        }
        Kind::BFloat16 => {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "Got unsupported ScalarType BFloat16".into(),
            });
        }
        _ => Tensor::from(value.double_value(&[]).ln()),
    })
}
