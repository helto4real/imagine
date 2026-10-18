//! Frozen Kitchen AWQ W4A16 wrappers; native kernels are kept unchanged.
use crate::{
    native::{Error, KernelArg, Kitchen},
    packing,
};
use tch::{Device, Kind, Tensor};

pub const DEFAULT_GROUP: i64 = 64;
// The executable reference says 256, although its docstring says 512.
const MMA_M_LIMIT: i64 = 256;

pub struct Linear<'a> {
    pub x: &'a Tensor,
    pub qweight: &'a Tensor,
    pub wscales: &'a Tensor,
    pub wzeros: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub group_size: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaRoute {
    Native,
    DequantThenMatmul,
}

fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}

fn flatten(x: &Tensor) -> Result<Tensor, Error> {
    let shape = x.size();
    let k = shape
        .last()
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    Ok(x.f_reshape([-1, *k])?)
}

fn check_group(k: i64, group: i64) -> Result<(), Error> {
    if group == 0 {
        return Err(exception("ZeroDivisionError", "integer modulo by zero"));
    }
    if k % group != 0 {
        return Err(exception(
            "ValueError",
            format!("K={k} not divisible by group_size={group}"),
        ));
    }
    Ok(())
}

fn check_width(k: i64, k_half: i64) -> Result<(), Error> {
    if k_half * 2 != k {
        return Err(exception(
            "ValueError",
            format!("qweight K//2={k_half} inconsistent with x K={k}"),
        ));
    }
    Ok(())
}

fn weight_shape(qweight: &Tensor) -> Result<(i64, i64), Error> {
    let shape = qweight.size();
    if shape.len() != 2 {
        return Err(exception(
            "ValueError",
            if shape.len() > 2 {
                "too many values to unpack (expected 2)".into()
            } else {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    shape.len()
                )
            },
        ));
    }
    Ok((shape[0], shape[1]))
}

fn restore(out: &Tensor, x: &Tensor, n: i64) -> Result<Tensor, Error> {
    let mut shape = x.size();
    shape.pop();
    shape.push(n);
    Ok(out.f_reshape(shape.as_slice())?)
}

fn weight(nibbles: &Tensor, request: &Linear<'_>, n: i64, k: i64) -> Result<Tensor, Error> {
    let w = nibbles
        .f_view_([n, k / request.group_size, request.group_size].as_slice())?
        .f_sub_scalar(8.0)?
        .f_mul(&request.wscales.f_tr()?.f_unsqueeze(-1)?)?
        .f_add(&request.wzeros.f_tr()?.f_unsqueeze(-1)?)?;
    Ok(w.f_view_([n, k].as_slice())?)
}

/// Original eager operation order, including out-of-place bias promotion.
pub fn eager(request: &Linear<'_>) -> Result<Tensor, Error> {
    let x = flatten(request.x)?;
    let k = x.size()[1];
    check_group(k, request.group_size)?;
    let (n, k_half) = weight_shape(request.qweight)?;
    check_width(k, k_half)?;
    let nibbles = packing::unpack_int4_row_major(request.qweight, false)?
        .f_to_kind(request.wscales.kind())?;
    let groups = nibbles.f_view_([n, k / request.group_size, request.group_size].as_slice())?;
    let scales = request.wscales.f_tr()?.f_unsqueeze(-1)?;
    let zeros = request.wzeros.f_tr()?.f_unsqueeze(-1)?;
    let w = groups
        .f_sub_scalar(8.0)?
        .f_mul(&scales)?
        .f_add(&zeros)?
        .f_view_([n, k].as_slice())?;
    let mut out = x.f_to_kind(request.wscales.kind())?.f_matmul(&w.f_tr()?)?;
    if let Some(bias) = request.bias {
        out = out.f_add(bias)?;
    }
    restore(&out, request.x, n)
}

/// Original CUDA wrapper on the initial SM89 platform.
/// # Safety
/// The native branch requires original CUDA row-major packed weight, scales,
/// zeros, matching FP16/BF16 storage, valid even group geometry and sufficient
/// capacities on the current device/stream. No implicit copies of weight,
/// scales or zeros are introduced. The ATen fallback retains its view rules.
pub unsafe fn cuda(native: &Kitchen, request: &Linear<'_>) -> Result<Tensor, Error> {
    unsafe { cuda_impl(native, request, None) }
}

/// Metadata-only development trace, with no hook in ordinary calls.
/// # Safety
/// Same native operand contract as cuda. Observer must not modify device,
/// stream, allocator or tensors, and must not retain additional tensors.
#[doc(hidden)]
pub unsafe fn cuda_observed(
    native: &Kitchen,
    request: &Linear<'_>,
    observer: &mut dyn FnMut(CudaRoute),
) -> Result<Tensor, Error> {
    unsafe { cuda_impl(native, request, Some(observer)) }
}

unsafe fn cuda_impl(
    native: &Kitchen,
    request: &Linear<'_>,
    observer: Option<&mut dyn FnMut(CudaRoute)>,
) -> Result<Tensor, Error> {
    let x = flatten(request.x)?;
    let (m, k) = (x.size()[0], x.size()[1]);
    let shape = request.qweight.size();
    let n = *shape
        .first()
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    check_group(k, request.group_size)?;
    let k_half = *shape
        .get(1)
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    check_width(k, k_half)?;
    let route;
    let mut out;
    if m > MMA_M_LIMIT {
        route = CudaRoute::DequantThenMatmul;
        let x = x.f_contiguous()?.f_to_kind(request.wscales.kind())?;
        let (n, k_half) = weight_shape(request.qweight)?;
        let k = k_half * 2;
        let x32 = request.qweight.f_to_kind(Kind::Int)?;
        let lo = x32.f_bitwise_and(15)?.f_to_kind(Kind::Int8)?;
        let hi = x32
            .f_bitwise_right_shift_tensor_scalar(4)?
            .f_bitwise_and(15)?
            .f_to_kind(Kind::Int8)?;
        let nibbles = Tensor::f_stack(&[lo, hi], -1)?
            .f_reshape([n, k])?
            .f_to_kind(request.wscales.kind())?;
        let w = weight(&nibbles, request, n, k)?;
        out = x.f_matmul(&w.f_tr()?)?;
    } else {
        route = CudaRoute::Native;
        out = Tensor::f_empty([m, n], (request.wscales.kind(), request.x.device()))?;
        let Device::Cuda(device) = request.x.device() else {
            return Err(Error::Native("CUDA AWQ requires CUDA input".into()));
        };
        let stream = native.current_stream(device)?;
        use KernelArg::{Int, Tensor as T};
        unsafe {
            native.call_unchecked(
                "awq_w4a16",
                &[
                    T(&x.f_contiguous()?.f_to_kind(request.wscales.kind())?),
                    T(&request.qweight.f_view_dtype(Kind::Uint8)?),
                    T(request.wscales),
                    T(request.wzeros),
                    T(&out),
                    Int(request.group_size),
                    Int(stream as i64),
                ],
            )?;
        }
    }
    if let Some(bias) = request.bias {
        let _ = out.f_add_(bias)?;
    }
    if let Some(observer) = observer {
        observer(route);
    }
    restore(&out, request.x, n)
}
