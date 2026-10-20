//! Inference adapter bypass algebra. Injection/owner lifecycles are separate
//! from these source h/g/forward operations; base weights are never merged here.
use super::{Adapter, Diagnostics, exception, gap};
use crate::{execution_value::Value, native::Error};
use std::sync::atomic::{AtomicBool, Ordering};
use tch::{Kind, Tensor};
#[derive(Clone)]
pub struct Convolution {
    pub dimension: usize,
    pub kernel_size: Vec<i64>,
    pub in_channels: Option<i64>,
    pub stride: Vec<i64>,
    pub padding: Vec<i64>,
    pub dilation: Vec<i64>,
    pub groups: i64,
}
#[derive(Clone)]
pub struct Settings {
    pub multiplier: f64,
    pub convolution: Option<Convolution>,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            multiplier: 1.,
            convolution: None,
        }
    }
}
fn raw_weight(adapter: &Adapter, index: usize) -> Result<Tensor, Error> {
    let value = adapter
        .weights
        .get(index)
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    value.as_tensor()
}
fn weight(adapter: &Adapter, index: usize, x: &Tensor) -> Result<Tensor, Error> {
    // Source .to(dtype=) does not implicitly move an un-injected adapter.
    Ok(raw_weight(adapter, index)?.f_to_kind(x.kind())?)
}
fn has(adapter: &Adapter, index: usize) -> bool {
    adapter
        .weights
        .get(index)
        .is_some_and(|v| !matches!(v, Value::None))
}
use super::adapter_scalar::{Rank, Scale, division_type_error, numeric, scale};
fn point_weight(w: &Tensor, dimension: usize) -> Result<Tensor, Error> {
    if w.dim() != 2 {
        return Ok(w.shallow_clone());
    }
    let mut shape = w.size();
    shape.extend(vec![1; dimension]);
    Ok(w.f_view(shape.as_slice())?)
}
fn input_weight(w: &Tensor, c: &Convolution) -> Result<Tensor, Error> {
    if w.dim() != 2 {
        return Ok(w.shallow_clone());
    }
    if let Some(channels) = c.in_channels {
        let mut shape = vec![w.size()[0], channels];
        shape.extend_from_slice(&c.kernel_size);
        Ok(w.f_view(shape.as_slice())?)
    } else {
        point_weight(w, c.dimension)
    }
}
fn op(x: &Tensor, w: &Tensor, conv: Option<&Convolution>, kwargs: bool) -> Result<Tensor, Error> {
    let Some(c) = conv else {
        return Ok(x.f_linear(w, None::<&Tensor>)?);
    };
    let (ones, zeros) = (vec![1; c.dimension], vec![0; c.dimension]);
    let (stride, padding, dilation, groups) = if kwargs {
        (&c.stride, &c.padding, &c.dilation, c.groups)
    } else {
        (&ones, &zeros, &ones, 1)
    };
    Ok(match c.dimension {
        1 => x.f_conv1d(w, None::<&Tensor>, stride, padding, dilation, groups)?,
        2 => x.f_conv2d(w, None::<&Tensor>, stride, padding, dilation, groups)?,
        3 => x.f_conv3d(w, None::<&Tensor>, stride, padding, dilation, groups)?,
        _ => return Err(gap("bypass module convolution dimension")),
    })
}
fn einsum(equation: &str, values: &[&Tensor]) -> Result<Tensor, Error> {
    Ok(Tensor::f_einsum(equation, values, None::<&[i64]>)?)
}
fn lora(adapter: &Adapter, x: &Tensor, settings: &Settings) -> Result<Tensor, Error> {
    let up = raw_weight(adapter, 0)?;
    let down = raw_weight(adapter, 1)?;
    let multiplier = scale(
        adapter,
        2,
        Some(Rank::Integer(i128::from(down.size()[0]))),
        settings.multiplier,
    )?;
    let mut up = up.f_to_kind(x.kind())?;
    let mut down = down.f_to_kind(x.kind())?;
    let mut mid = if has(adapter, 3) {
        Some(weight(adapter, 3, x)?)
    } else {
        None
    };
    if let Some(c) = &settings.convolution {
        down = input_weight(&down, c)?;
        up = point_weight(&up, c.dimension)?;
        if let Some(value) = &mid {
            mid = Some(point_weight(value, c.dimension)?);
        }
    }
    let conv = settings.convolution.as_ref();
    let out = if let Some(mid) = mid {
        let hidden = op(x, &down, conv, false)?;
        let hidden = op(&hidden, &mid, conv, true)?;
        op(&hidden, &up, conv, false)?
    } else {
        let hidden = op(x, &down, conv, true)?;
        op(&hidden, &up, conv, false)?
    };
    multiplier.multiply(&out)
}
static LOHA_WARNED: AtomicBool = AtomicBool::new(false);
/// Native counterpart of _warn_loha_bypass_inefficient.cache_clear().
/// Ordinary inference never clears this process-local source cache implicitly.
pub fn clear_loha_warning_cache() {
    LOHA_WARNED.store(false, Ordering::Relaxed);
}
fn loha(
    adapter: &Adapter,
    x: &Tensor,
    settings: &Settings,
    diagnostics: &mut Diagnostics,
) -> Result<Tensor, Error> {
    if !LOHA_WARNED.swap(true, Ordering::Relaxed) {
        diagnostics.warnings.push("LoHa bypass mode is inefficient: full weight diff is computed each forward pass. Consider using LoRA or LoKr for training with bypass mode.".into());
    }
    let multiplier = scale(
        adapter,
        2,
        Some(Rank::Integer(i128::from(raw_weight(adapter, 1)?.size()[0]))),
        settings.multiplier,
    )?;
    let (w1a, w1b, w2a, w2b) = (
        weight(adapter, 0, x)?,
        weight(adapter, 1, x)?,
        weight(adapter, 3, x)?,
        weight(adapter, 4, x)?,
    );
    let (m1, m2) = if has(adapter, 5) && has(adapter, 6) {
        (
            einsum(
                "i j k l, j r, i p -> p r k l",
                &[&weight(adapter, 5, x)?, &w1b, &w1a],
            )?,
            einsum(
                "i j k l, j r, i p -> p r k l",
                &[&weight(adapter, 6, x)?, &w2b, &w2a],
            )?,
        )
    } else {
        (w1a.f_matmul(&w1b)?, w2a.f_matmul(&w2b)?)
    };
    let mut diff = multiplier.multiply(&m1.f_mul(&m2)?)?;
    if let Some(c) = &settings.convolution {
        diff = input_weight(&diff, c)?;
    }
    op(x, &diff, settings.convolution.as_ref(), true)
}
fn lokr(adapter: &Adapter, x: &Tensor, settings: &Settings) -> Result<Tensor, Error> {
    let (use_w1, use_w2, tucker) = (has(adapter, 0), has(adapter, 1), has(adapter, 7));
    let multiplier = if use_w1 && use_w2 {
        let alpha = &adapter.weights[2];
        if matches!(alpha, Value::None) {
            Scale::Scalar(settings.multiplier)
        } else if matches!(alpha, Value::Tensor(_)) {
            let alpha = alpha.as_tensor()?;
            Scale::Tensor(alpha.f_div(&alpha)?.f_mul_scalar(settings.multiplier)?)
        } else {
            let rank = Rank::from_value(alpha).ok_or_else(|| {
                division_type_error(
                    alpha,
                    super::options::Options::from_value(alpha.clone()).typename(),
                )
            })?;
            scale(adapter, 2, Some(rank), settings.multiplier)?
        }
    } else {
        let rank = if !use_w1 {
            Some(Rank::Integer(i128::from(
                adapter.weights[4].as_tensor()?.size()[0],
            )))
        } else {
            Some(Rank::Integer(i128::from(
                adapter.weights[6].as_tensor()?.size()[0],
            )))
        };
        scale(adapter, 2, rank, settings.multiplier)?
    };
    let c = if use_w1 {
        weight(adapter, 0, x)?
    } else {
        weight(adapter, 3, x)?.f_matmul(&weight(adapter, 4, x)?)?
    };
    let uq = c.size()[1];
    let conv = settings.convolution.as_ref();
    let mut a = None;
    let mut b = None;
    let mut ba = None;
    if use_w2 {
        ba = Some(weight(adapter, 1, x)?);
    } else {
        let mut wa = weight(adapter, 6, x)?;
        let mut wb = weight(adapter, 5, x)?;
        if let Some(conv) = conv {
            if tucker {
                wa = point_weight(&wa, conv.dimension)?;
            }
            wb = point_weight(&wb, conv.dimension)?;
        }
        a = Some(wa);
        b = Some(wb);
    }
    let shape = x.size();
    let input = if conv.is_some() {
        let mut s = vec![shape[0] * uq, -1];
        s.extend_from_slice(&shape[2..]);
        x.f_reshape(&s)?
    } else {
        let mut s = shape[..shape.len() - 1].to_vec();
        s.extend([uq, -1]);
        x.f_reshape(&s)?
    };
    let hb = if let Some(ba) = ba {
        op(&input, &ba, conv, true)?
    } else {
        let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
        if let Some(conv) = conv {
            if tucker {
                let t = point_weight(&weight(adapter, 7, x)?, conv.dimension)?;
                let ha = op(&input, a, Some(conv), false)?;
                let ht = op(&ha, &t, Some(conv), true)?;
                op(&ht, b, Some(conv), false)?
            } else {
                let ha = op(&input, a, Some(conv), true)?;
                op(&ha, b, Some(conv), false)?
            }
        } else {
            let ha = op(&input, a, None, false)?;
            op(&ha, b, None, false)?
        }
    };
    let cross = if conv.is_some() {
        let mut s = vec![shape[0], -1];
        s.extend_from_slice(&hb.size()[1..]);
        hb.f_view(s.as_slice())?.f_transpose(1, -1)?
    } else {
        hb.f_transpose(-1, -2)?
    };
    let hc = cross.f_linear(&c, None::<&Tensor>)?;
    let out = if conv.is_some() {
        let hc = hc.f_transpose(1, -1)?;
        let mut s = vec![shape[0], -1];
        s.extend_from_slice(&hc.size()[3..]);
        hc.f_reshape(&s)?
    } else {
        let hc = hc.f_transpose(-1, -2)?;
        let mut s = hc.size()[..hc.dim() - 2].to_vec();
        s.push(-1);
        hc.f_reshape(&s)?
    };
    multiplier.multiply(&out)
}
fn glora_paths(
    adapter: &Adapter,
    x: &Tensor,
    settings: &Settings,
) -> Result<(Tensor, Tensor), Error> {
    let (mut a1, mut a2, mut b1, mut b2) = (
        weight(adapter, 0, x)?,
        weight(adapter, 1, x)?,
        weight(adapter, 2, x)?,
        weight(adapter, 3, x)?,
    );
    let mut old = false;
    let mut rank = None;
    if b2.size()[1] == b1.size()[0] && b1.size()[0] == a1.size()[0] && a1.size()[0] == a2.size()[1]
    {
        old = true;
        rank = Some(Rank::Integer(i128::from(a1.size()[0])));
    }
    if b2.size()[0] == b1.size()[1]
        && b1.size()[1] == a1.size()[1]
        && a1.size()[1] == a2.size()[0]
        && !(old && a2.size()[0] == *x.size().last().unwrap())
    {
        old = false;
        rank = Some(Rank::Integer(i128::from(a2.size()[0])));
    }
    let multiplier = scale(adapter, 4, rank, settings.multiplier)?;
    let (a, b) = if let Some(c) = &settings.convolution {
        a1 = point_weight(&a1, c.dimension)?;
        a2 = point_weight(&a2, c.dimension)?;
        b1 = point_weight(&b1, c.dimension)?;
        b2 = input_weight(&b2, c)?;
        let a = multiplier.multiply(&op(&op(x, &a2, Some(c), false)?, &a1, Some(c), false)?)?;
        // Source GLoRA passes only stride/padding on its B path, even if the
        // original module supplied dilation/groups in kw_dict.
        let mut b_config = c.clone();
        b_config.dilation = vec![1; c.dimension];
        b_config.groups = 1;
        let b = multiplier.multiply(&op(
            &op(x, &b2, Some(&b_config), true)?,
            &b1,
            Some(c),
            false,
        )?)?;
        (a, b)
    } else if old {
        (
            multiplier.multiply(&op(&op(x, &a2, None, false)?, &a1, None, false)?)?,
            multiplier.multiply(&op(&op(x, &b1, None, false)?, &b2, None, false)?)?,
        )
    } else {
        (
            multiplier.multiply(&op(&op(x, &a1, None, false)?, &a2, None, false)?)?,
            multiplier.multiply(&op(&op(x, &b2, None, false)?, &b1, None, false)?)?,
        )
    };
    Ok((a, b))
}
fn orthogonal(
    adapter: &Adapter,
    y: &Tensor,
    settings: &Settings,
    butterfly: bool,
) -> Result<Tensor, Error> {
    let blocks =
        adapter.weights[0]
            .as_tensor()?
            .f_to_device_(y.device(), y.kind(), false, false)?;
    let size = blocks.size();
    let block_size = if butterfly { size[2] } else { size[1] };
    let eye = Tensor::f_eye(block_size, (y.kind(), y.device()))?;
    let q = blocks.f_sub(&blocks.f_transpose(-1, -2)?)?;
    let alpha = match &adapter.weights[2] {
        Value::None => Scale::Scalar(0.),
        value @ Value::Tensor(_) => Scale::Tensor(value.as_tensor()?),
        value => Scale::Scalar(numeric(value).ok_or_else(|| {
            exception(
                "TypeError",
                format!(
                    "'>' not supported between instances of '{}' and 'int'",
                    super::options::Options::from_value(value.clone()).typename()
                ),
            )
        })?),
    };
    let normalized = if alpha.positive()? {
        let norm = q.f_norm()?.f_add_scalar(1e-8)?;
        if alpha.less_than(&norm)? {
            alpha.multiply(&q)?.f_div(&norm)?
        } else {
            q.shallow_clone()
        }
    } else {
        q.shallow_clone()
    };
    let rotations = eye.f_add(&normalized)?.f_matmul(
        &eye.f_sub(&normalized)?
            .f_to_kind(Kind::Float)?
            .f_inverse()?,
    )?;
    let is_conv = settings.convolution.is_some();
    let y = if is_conv {
        y.f_transpose(1, -1)?
    } else {
        y.shallow_clone()
    };
    let mut out = if butterfly {
        let mut input = y.shallow_clone();
        let half = block_size / 2;
        for i in 0..size[0] {
            let mut rotation = rotations.f_get(i)?;
            if settings.multiplier != 1. {
                rotation = rotation
                    .f_mul_scalar(settings.multiplier)?
                    .f_add(&eye.f_mul_scalar(1. - settings.multiplier)?)?;
            }
            let k = (1_i64 << i) * half;
            input = input
                .f_unflatten(-1, [-1, 2, k])?
                .f_transpose(-2, -1)?
                .f_flatten(-3, -1)?
                .f_unflatten(-1, [-1, block_size])?;
            input = einsum("b i j, ... b j -> ... b i", &[&rotation, &input])?;
            input = input
                .f_flatten(-2, -1)?
                .f_unflatten(-1, [-1, k, 2])?
                .f_transpose(-2, -1)?
                .f_flatten(-3, -1)?;
        }
        input
    } else {
        let rotation = rotations
            .f_mul_scalar(settings.multiplier)?
            .f_add(&eye.f_mul_scalar(1. - settings.multiplier)?)?;
        let mut shape = y.size();
        let last = shape.pop().unwrap();
        let mut blocked = shape.clone();
        blocked.extend([size[0], block_size]);
        let out = einsum(
            "k n m, ... k n -> ... k m",
            &[&rotation, &y.f_view(blocked.as_slice())?],
        )?;
        shape.push(last);
        out.f_view(shape.as_slice())?
    };
    if has(adapter, 1) {
        let scale =
            adapter.weights[1]
                .as_tensor()?
                .f_to_device_(y.device(), y.kind(), false, false)?;
        out = out.f_mul(&if butterfly {
            scale.f_transpose(0, -1)?
        } else {
            scale.f_view([-1])?
        })?;
    }
    if is_conv {
        out = out.f_transpose(1, -1)?;
    }
    Ok(out)
}
pub fn additive(
    adapter: &Adapter,
    x: &Tensor,
    base: &Tensor,
    settings: &Settings,
    diagnostics: &mut Diagnostics,
) -> Result<Tensor, Error> {
    match adapter.name.as_str() {
        "lora" => lora(adapter, x, settings),
        "loha" => loha(adapter, x, settings, diagnostics),
        "lokr" => lokr(adapter, x, settings),
        "glora" => glora_paths(adapter, x, settings).map(|(_, b)| b),
        "oft" | "boft" => Ok(base.f_zeros_like()?),
        _ => Err(gap(format!("adapter bypass {}", adapter.name))),
    }
}
pub fn transform(adapter: &Adapter, y: &Tensor, settings: &Settings) -> Result<Tensor, Error> {
    match adapter.name.as_str() {
        "oft" => orthogonal(adapter, y, settings, false),
        "boft" => orthogonal(adapter, y, settings, true),
        _ => Ok(y.shallow_clone()),
    }
}
pub fn forward(
    adapter: &Adapter,
    x: &Tensor,
    settings: &Settings,
    original: &mut dyn FnMut(&Tensor) -> Result<Tensor, Error>,
    diagnostics: &mut Diagnostics,
) -> Result<Tensor, Error> {
    if adapter.name == "glora" {
        let (a, b) = glora_paths(adapter, x, settings)?;
        return Ok(original(&x.f_add(&a)?)?.f_add(&b)?);
    }
    let base = original(x)?;
    let delta = additive(adapter, x, &base, settings, diagnostics)?;
    transform(adapter, &base.f_add(&delta)?, settings)
}
