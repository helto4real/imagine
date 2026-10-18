//! Qwen21 reference geometry/alpha and original Pillow Lanczos via a C-only ABI.
use crate::native::Error;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
};
use tch::{Device, Kind, Tensor};
type Resize = unsafe extern "C" fn(*const u8, i32, i32, i32, i32, i32, *mut u8) -> i32;
type LastError = unsafe extern "C" fn() -> *const c_char;
pub struct Pillow {
    handle: NonNull<c_void>,
    resize: Resize,
    error: LastError,
}
impl Drop for Pillow {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle.as_ptr());
        }
    }
}
impl Pillow {
    /// # Safety
    /// The selected file is the verified matching source-built buffer-only ABI.
    pub unsafe fn open(path: &Path) -> Result<Self, Error> {
        let path =
            CString::new(path.as_os_str().as_bytes()).map_err(|e| Error::Native(e.to_string()))?;
        let handle =
            NonNull::new(unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) })
                .ok_or_else(|| Error::Native("Cannot load verified Pillow buffer ABI".into()))?;
        let symbol = |name: &CStr| unsafe { libc::dlsym(handle.as_ptr(), name.as_ptr()) };
        let resize = symbol(c"imagine_pillow_resize_u8");
        let error = symbol(c"imagine_pillow_error");
        let abi = symbol(c"imagine_pillow_abi");
        if resize.is_null() || error.is_null() || abi.is_null() {
            unsafe {
                libc::dlclose(handle.as_ptr());
            }
            return Err(Error::Native("Incomplete Pillow buffer ABI".into()));
        }
        let abi: unsafe extern "C" fn() -> i32 = unsafe { std::mem::transmute(abi) };
        if unsafe { abi() } != 1 {
            unsafe {
                libc::dlclose(handle.as_ptr());
            }
            return Err(Error::Native("Pillow buffer ABI version mismatch".into()));
        }
        Ok(Self {
            handle,
            resize: unsafe { std::mem::transmute::<*mut c_void, Resize>(resize) },
            error: unsafe { std::mem::transmute::<*mut c_void, LastError>(error) },
        })
    }
    pub fn resize_rgb(
        &self,
        bytes: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        out_width: i32,
        out_height: i32,
    ) -> Result<Vec<u8>, Error> {
        if width <= 0 || height <= 0 || out_width <= 0 || out_height <= 0 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "height and width must be > 0".into(),
            });
        }
        if ![3, 4].contains(&channels) {
            return Err(Error::BindingGap {
                backend: "pillow_buffer".into(),
                operation: "Only Qwen RGB/RGBA buffer binding is registered".into(),
            });
        }
        let length = |w: i32, h: i32| {
            (w as usize)
                .checked_mul(h as usize)
                .and_then(|n| n.checked_mul(channels as usize))
                .ok_or_else(|| Error::Native("Image buffer capacity overflow".into()))
        };
        if bytes.len() != length(width, height)? {
            return Err(Error::Native(
                "Invalid trusted Pillow input buffer length".into(),
            ));
        }
        let mut out = vec![0; length(out_width, out_height)?];
        let result = unsafe {
            (self.resize)(
                bytes.as_ptr(),
                width,
                height,
                channels,
                out_width,
                out_height,
                out.as_mut_ptr(),
            )
        };
        if result != 0 {
            return Err(Error::ReferenceException {
                class: if result == 2 {
                    "MemoryError"
                } else {
                    "ValueError"
                },
                message: unsafe { CStr::from_ptr((self.error)()) }
                    .to_string_lossy()
                    .into_owned(),
            });
        }
        Ok(out)
    }
    /// Original comfy.utils.lanczos on NCHW RGB/RGBA tensors. NumPy rounds
    /// multiplication in the incoming dtype, truncates clipped bytes, then
    /// converts resized bytes through Float32 / 255 before moving/casting back.
    pub fn lanczos(&self, samples: &Tensor, width: i64, height: i64) -> Result<Tensor, Error> {
        if samples.kind() == Kind::BFloat16 {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "Got unsupported ScalarType BFloat16".into(),
            });
        }
        let shape = samples.size();
        let channels = shape[1];
        let mut outputs = vec![];
        for i in 0..shape[0] {
            let x = samples
                .f_get(i)?
                .f_permute([1, 2, 0])?
                .f_to_device(Device::Cpu)?
                .f_mul_scalar(255.)?
                .f_clamp(0., 255.)?
                .f_to_kind(Kind::Uint8)?
                .f_contiguous()?;
            let mut bytes = vec![0; x.numel()];
            let n = bytes.len();
            x.f_copy_data_u8(&mut bytes, n)?;
            let int = |n: i64| {
                i32::try_from(n).map_err(|_| Error::ReferenceException {
                    class: "OverflowError",
                    message: "signed integer is greater than maximum".into(),
                })
            };
            let resized = self.resize_rgb(
                &bytes,
                int(shape[3])?,
                int(shape[2])?,
                int(channels)?,
                int(width)?,
                int(height)?,
            )?;
            let values = resized
                .into_iter()
                .map(|n| n as f32 / 255_f32)
                .collect::<Vec<_>>();
            outputs.push(
                Tensor::f_from_slice(&values)?
                    .f_view([height, width, channels])?
                    .f_permute([2, 0, 1])?,
            );
        }
        Ok(Tensor::f_stack(&outputs, 0)?.f_to_device_(
            samples.device(),
            samples.kind(),
            false,
            false,
        )?)
    }
}
pub type ReferenceVaeCallback<'a> = dyn FnMut(&Tensor) -> Result<Tensor, Error> + 'a;
pub struct ReferenceImages {
    pub vision: Vec<Tensor>,
    pub reference_latents: Vec<Tensor>,
    pub width: i64,
    pub height: i64,
}
/// Qwen21 VAE.vae_encode_crop_pixels: centered spatial compression crop,
/// four output channels and opaque alpha padding, before NCHW/process_input.
pub fn vae_crop_pixels(pixels: &Tensor) -> Result<Tensor, Error> {
    let shape = pixels.size();
    let mut out = pixels.shallow_clone();
    for (axis, &dim) in shape.iter().enumerate().take(shape.len() - 1).skip(1) {
        let size = dim / 16 * 16;
        if size != dim {
            out = out.f_narrow(axis as i64, dim % 16 / 2, size)?;
        }
    }
    let channels = *shape.last().unwrap();
    if channels > 4 {
        out = out.f_slice(-1, 0, 4, 1)?;
    } else if channels < 4 {
        out = out.f_pad([0, 4 - channels], "constant", Some(1.))?;
    }
    Ok(out)
}
/// Original default VAE input transform runs before the BF16/device cast.
pub fn vae_encode_pixels(pixels: &Tensor, device: Device) -> Result<Tensor, Error> {
    Ok(vae_crop_pixels(pixels)?
        .f_movedim_int(-1, 1)?
        .f_mul_scalar(2.)?
        .f_sub_scalar(1.)?
        .f_to_kind(Kind::BFloat16)?
        .f_to_device(device)?)
}
/// Original TextEncodeQwenImage21.execute image loop, after numeric input sorting.
/// Ordered inputs are tensors, not ComfyUI graphs or UI nodes.
pub fn prepare_reference_images(
    pillow: &Pillow,
    ordered: &[Tensor],
    resolution: i64,
    mut encode: Option<&mut ReferenceVaeCallback<'_>>,
) -> Result<ReferenceImages, Error> {
    let mut out = ReferenceImages {
        vision: vec![],
        reference_latents: vec![],
        width: if resolution == 0 { 1024 } else { resolution },
        height: if resolution == 0 { 1024 } else { resolution },
    };
    for image in ordered {
        let samples = image.f_slice(0, 0, 1, 1)?.f_permute([0, 3, 1, 2])?;
        let shape = samples.size();
        let (width, height) = if resolution > 0 {
            let ratio = shape[3] as f64 / shape[2] as f64;
            let area = (resolution * resolution) as f64;
            (
                ((area * ratio).sqrt() / 32.).round_ties_even() as i64 * 32,
                ((area / ratio).sqrt() / 32.).round_ties_even() as i64 * 32,
            )
        } else {
            (
                (shape[3] as f64 / 32.).round_ties_even() as i64 * 32,
                (shape[2] as f64 / 32.).round_ties_even() as i64 * 32,
            )
        };
        let (width, height) = (width.max(32), height.max(32));
        let s = if (width, height) == (shape[3], shape[2]) {
            image.f_slice(0, 0, 1, 1)?
        } else {
            pillow
                .lanczos(&samples, width, height)?
                .f_permute([0, 2, 3, 1])?
        };
        if out.vision.is_empty() {
            out.width = width;
            out.height = height;
        }
        let mut rgb = s.f_slice(-1, 0, 3, 1)?;
        if s.size()[3] > 3 {
            let alpha = s.f_slice(-1, 3, i64::MAX, 1)?;
            rgb = rgb.f_mul(&alpha)?.f_add(&alpha.f_rsub_scalar(1.)?)?;
        }
        out.vision.push(rgb);
        if let Some(encode) = encode.as_mut() {
            out.reference_latents.push(encode(&s)?);
        }
    }
    Ok(out)
}
