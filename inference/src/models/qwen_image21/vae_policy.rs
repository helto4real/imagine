//! Qwen's image-only `sd.VAE` policy and `utils.tiled_scale_multidim`.
//! Model calls and memory services stay injectable so failure tests allocate
//! only small tensors, never full trained weights or an intentional OOM.
use crate::native::Error;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct ImageTiles {
    pub x: i64,
    pub y: i64,
    pub overlap: i64,
}
impl ImageTiles {
    pub fn decode() -> Self {
        Self {
            x: 64,
            y: 64,
            overlap: 16,
        }
    }
    pub fn encode() -> Self {
        Self {
            x: 512,
            y: 512,
            overlap: 64,
        }
    }
}
#[derive(Clone, Copy, Default)]
pub struct ImageTileOptions {
    pub x: Option<i64>,
    pub y: Option<i64>,
    pub overlap: Option<i64>,
}

pub struct ImageScale {
    pub tiles: ImageTiles,
    pub factor: f64,
    pub channels: i64,
    pub output_device: Device,
    /// torch.empty/ones/zeros use the current Torch default dtype, rather
    /// than the input or the VAE's working dtype.
    pub default_kind: Kind,
}
fn scaled(n: i64, factor: f64) -> i64 {
    (n as f64 * factor).round_ties_even() as i64
}
fn positions(size: i64, tile: i64, overlap: i64) -> Result<Vec<i64>, Error> {
    if size <= tile {
        return Ok(vec![0]);
    }
    let step = tile - overlap;
    if step == 0 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "range() arg 3 must not be zero".into(),
        });
    }
    let end = size - overlap;
    let mut values = Vec::new();
    let mut at = 0;
    while if step > 0 { at < end } else { at > end } {
        values.push(at);
        at += step;
    }
    Ok(values)
}
pub fn tiled_scale_image(
    samples: &Tensor,
    scale: ImageScale,
    mut function: impl FnMut(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let _inference = crate::native::inference_mode(true)?;
    let ImageScale {
        tiles,
        factor,
        channels,
        output_device,
        default_kind,
    } = scale;
    let shape = samples.size();
    let opts = (default_kind, output_device);
    let output = Tensor::f_empty(
        [
            shape[0],
            channels,
            scaled(shape[2], factor),
            scaled(shape[3], factor),
        ],
        opts,
    )?;
    for b in 0..shape[0] {
        let input = samples.f_narrow(0, b, 1)?;
        let mut out = output.f_narrow(0, b, 1)?;
        if shape[2] <= tiles.y && shape[3] <= tiles.x {
            out.f_copy_(&function(&input)?.f_to_device(output_device)?)?;
            continue;
        }
        let _ = out.f_zero_()?;
        let divisor = Tensor::f_zeros([1, 1, out.size()[2], out.size()[3]], opts)?;
        for y in positions(shape[2], tiles.y, tiles.overlap)? {
            for x in positions(shape[3], tiles.x, tiles.overlap)? {
                let at = [y, x];
                let sizes = [tiles.y, tiles.x];
                let mut tile = input.shallow_clone();
                let mut destination = [0; 2];
                for d in 0..2 {
                    let pos = at[d].min(shape[d + 2] - tiles.overlap).max(0);
                    tile = tile.f_narrow(d as i64 + 2, pos, sizes[d].min(shape[d + 2] - pos))?;
                    destination[d] = scaled(pos, factor);
                }
                let mut result = function(&tile)?.f_to_device(output_device)?;
                let mut mask = Tensor::f_ones([1, 1, result.size()[2], result.size()[3]], opts)?;
                for d in 2..4 {
                    let feather = scaled(tiles.overlap, factor);
                    if feather >= mask.size()[d] {
                        continue;
                    }
                    for t in 0..feather {
                        let a = (t + 1) as f64 / feather as f64;
                        let _ = mask.f_narrow(d as i64, t, 1)?.f_mul_scalar_(a)?;
                        let _ = mask
                            .f_narrow(d as i64, mask.size()[d] - 1 - t, 1)?
                            .f_mul_scalar_(a)?;
                    }
                }
                let mut dst = out.shallow_clone();
                let mut div = divisor.shallow_clone();
                for (d, pos) in destination.into_iter().enumerate() {
                    let axis = d as i64 + 2;
                    let length = result.size()[d + 2].min(dst.size()[d + 2] - pos);
                    dst = dst.f_narrow(axis, pos, length)?;
                    div = div.f_narrow(axis, pos, length)?;
                    if length < result.size()[d + 2] {
                        result = result.f_narrow(axis, 0, length)?;
                        mask = mask.f_narrow(axis, 0, length)?;
                    }
                }
                let _ = dst.f_add_(&result.f_mul(&mask)?)?;
                let _ = div.f_add_(&mask)?;
            }
        }
        let _ = out.f_div_(&divisor)?;
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeOperation {
    Encode,
    Decode,
}
pub trait ImageVaeServices {
    fn load(&mut self, memory_required: f64) -> Result<(), Error>;
    fn free_memory(&mut self) -> Result<f64, Error>;
    fn empty_cache(&mut self) -> Result<(), Error>;
    fn call(&mut self, operation: VaeOperation, input: &Tensor) -> Result<Tensor, Error>;
    /// Source `is_oom`, including clearing a pending CUDA asynchronous error.
    /// A plain RuntimeError with an OOM-looking message must not be swallowed.
    fn is_oom(&mut self, error: &Error) -> Result<bool, Error>;
}
/// Frozen CUDA `model_management.is_oom`, without classifying ordinary error
/// strings as allocation failures. The asynchronous-error clearing uses the
/// same two initialized byte tensors as the original helper.
pub fn image_vae_is_oom(
    native: &crate::native::Kitchen,
    device: Device,
    error: &Error,
) -> Result<bool, Error> {
    let mut bindings = crate::memory::exceptions::NativeOomBindings(|| {
        crate::memory::exceptions::discard_native_cuda_async_error(native, device)
    });
    crate::memory::exceptions::is_oom(error, &mut bindings)
}
#[derive(Clone, Copy)]
pub enum ImageVaeMemory {
    Wan22Image,
    /// AutoencoderKL's original estimate, with Flux.2's latent BatchNorm
    /// multiplier on decode. The AMD ratio is a source hardware predicate.
    Flux2 {
        amd: bool,
        /// ComfyUI d07dff45: the fused strip decode needs no 4x BatchNorm-latent
        /// multiplier.
        fused: bool,
    },
}
impl ImageVaeMemory {
    pub fn required(self, operation: VaeOperation, shape: &[i64], dtype: Kind) -> f64 {
        match self {
            Self::Wan22Image => {
                let factor = match operation {
                    VaeOperation::Encode => 600,
                    VaeOperation::Decode => 900 * 16 * 16,
                };
                (factor * shape[2] * shape[3]) as f64 * dtype.elt_size_in_bytes() as f64
            }
            Self::Flux2 { amd, fused } => {
                let ratio = if amd { 2.73 } else { 1. };
                match operation {
                    VaeOperation::Encode => {
                        (1767 * shape[2] * shape[3]) as f64
                            * dtype.elt_size_in_bytes() as f64
                            * ratio
                    }
                    VaeOperation::Decode => {
                        (2178 * shape[2] * shape[3] * 64) as f64
                            * dtype.elt_size_in_bytes() as f64
                            * ratio
                            * if fused { 1. } else { 4. }
                    }
                }
            }
        }
    }
}
#[derive(Clone, Copy)]
pub struct ImageVaePolicy {
    pub device: Device,
    pub dtype: Kind,
    pub output_device: Device,
    pub output_kind: Kind,
    pub default_kind: Kind,
    pub output_channels: i64,
    pub latent_channels: i64,
}
impl ImageVaePolicy {
    /// Explicit VAE.encode_tiled/decode_tiled image entry point. Encode input
    /// has already been source-cropped and moved from NHWC to BCHW.
    pub fn execute_tiled<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        input: &Tensor,
        options: ImageTileOptions,
        services: &mut S,
    ) -> Result<Tensor, Error> {
        self.execute_tiled_with_memory(op, input, options, services, ImageVaeMemory::Wan22Image)
    }
    pub fn execute_tiled_with_memory<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        input: &Tensor,
        options: ImageTileOptions,
        services: &mut S,
        memory: ImageVaeMemory,
    ) -> Result<Tensor, Error> {
        let mut memory_shape = input.size();
        if op == VaeOperation::Decode {
            // _tile_bounded_shape clamps only explicitly supplied dimensions;
            // None retains the full estimate even though the tiler has defaults.
            if let Some(y) = options.y {
                memory_shape[2] = memory_shape[2].min(y);
            }
            if let Some(x) = options.x {
                memory_shape[3] = memory_shape[3].min(x);
            }
        }
        services.load(memory.required(op, &memory_shape, self.dtype))?;
        let default = if op == VaeOperation::Encode {
            ImageTiles::encode()
        } else {
            ImageTiles::decode()
        };
        let output = self.tiled(
            op,
            input,
            ImageTiles {
                x: options.x.unwrap_or(default.x),
                y: options.y.unwrap_or(default.y),
                overlap: options.overlap.unwrap_or(default.overlap),
            },
            services,
        )?;
        Ok(if op == VaeOperation::Decode {
            output.f_movedim_int(1, -1)?
        } else {
            output
        })
    }
    pub fn memory_used(&self, operation: VaeOperation, shape: &[i64]) -> f64 {
        ImageVaeMemory::Wan22Image.required(operation, shape, self.dtype)
    }
    fn process_output(&self, mut x: Tensor) -> Result<Tensor, Error> {
        super::vae_process_output(&mut x)?;
        Ok(x)
    }
    fn run<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        x: &Tensor,
        services: &mut S,
        tiled: bool,
    ) -> Result<Tensor, Error> {
        let x = match op {
            VaeOperation::Encode => x
                .f_mul_scalar(2.)?
                .f_sub_scalar(1.)?
                .f_to_kind(self.dtype)?
                .f_to_device(self.device)?,
            VaeOperation::Decode => x.f_to_kind(self.dtype)?.f_to_device(self.device)?,
        };
        let y = services.call(op, &x)?;
        if tiled {
            return Ok(y.f_to_kind(self.output_kind)?);
        }
        Ok(match op {
            VaeOperation::Encode => y
                .f_to_device(self.output_device)?
                .f_to_kind(self.output_kind)?,
            VaeOperation::Decode => {
                y.f_to_device_(self.output_device, self.output_kind, false, true)?
            }
        })
    }
    /// Input is source-cropped BCHW, before process_input. Decode input is BCHW
    /// or BCTHW (the image VAE selects the first temporal slice).
    pub fn execute<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        input: &Tensor,
        services: &mut S,
    ) -> Result<Tensor, Error> {
        self.execute_with_memory(op, input, services, ImageVaeMemory::Wan22Image)
    }
    pub fn execute_with_memory<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        input: &Tensor,
        services: &mut S,
        estimate: ImageVaeMemory,
    ) -> Result<Tensor, Error> {
        let input = if op == VaeOperation::Decode && input.dim() == 5 {
            input.f_select(2, 0)?
        } else {
            input.shallow_clone()
        };
        // Keep the failed attempt's temporaries out of scope before empty_cache
        // and the tiled retry, like the source's post-except do_tile branch.
        let attempt = (|| {
            let memory = estimate.required(op, &input.size(), self.dtype);
            services.load(memory)?;
            let divisor = if op == VaeOperation::Encode {
                memory.max(1.)
            } else {
                memory
            };
            let batch = ((services.free_memory()? / divisor) as i64).max(1);
            let mut output: Option<Tensor> = None;
            let mut start = 0;
            while start < input.size()[0] {
                let length = batch.min(input.size()[0] - start);
                let result = self.run(op, &input.f_narrow(0, start, length)?, services, false)?;
                if output.is_none() {
                    let mut shape = result.size();
                    shape[0] = input.size()[0];
                    output = Some(Tensor::f_empty(
                        &shape,
                        (self.output_kind, self.output_device),
                    )?);
                }
                let mut dst = output.as_ref().unwrap().f_narrow(0, start, length)?;
                dst.f_copy_(&result)?;
                if op == VaeOperation::Decode {
                    let _ = self.process_output(dst)?;
                }
                start += length;
            }
            output.ok_or_else(|| Error::BindingGap {
                backend: "image_vae".into(),
                operation: "empty batch result".into(),
            })
        })();
        match attempt {
            Ok(output) => Ok(if op == VaeOperation::Decode {
                output.f_movedim_int(1, -1)?
            } else {
                output
            }),
            Err(error) => {
                if !services.is_oom(&error)? {
                    return Err(error);
                }
                drop(error);
                services.empty_cache()?;
                let out = self.tiled(
                    op,
                    &input,
                    match op {
                        VaeOperation::Encode => ImageTiles::encode(),
                        VaeOperation::Decode => ImageTiles::decode(),
                    },
                    services,
                )?;
                Ok(if op == VaeOperation::Decode {
                    out.f_movedim_int(1, -1)?
                } else {
                    out
                })
            }
        }
    }
    pub fn tiled<S: ImageVaeServices>(
        &self,
        op: VaeOperation,
        input: &Tensor,
        tile: ImageTiles,
        services: &mut S,
    ) -> Result<Tensor, Error> {
        // Original progress-step arithmetic precedes all three model passes;
        // preserve its ZeroDivisionError even when a later pass would fit.
        for (x, y) in [
            (tile.x, tile.y),
            (tile.x / 2, tile.y * 2),
            (tile.x * 2, tile.y / 2),
        ] {
            if (input.size()[2] > y && y == tile.overlap)
                || (input.size()[3] > x && x == tile.overlap)
            {
                return Err(Error::ReferenceException {
                    class: "ZeroDivisionError",
                    message: "division by zero".into(),
                });
            }
        }
        let aspect = |x, y| ImageTiles {
            x,
            y,
            overlap: tile.overlap,
        };
        let shapes = match op {
            VaeOperation::Encode => [
                tile,
                aspect(tile.x * 2, tile.y / 2),
                aspect(tile.x / 2, tile.y * 2),
            ],
            VaeOperation::Decode => [
                aspect(tile.x / 2, tile.y * 2),
                aspect(tile.x * 2, tile.y / 2),
                tile,
            ],
        };
        let mut outputs = shapes.into_iter().map(|tiles| {
            tiled_scale_image(
                input,
                ImageScale {
                    tiles,
                    factor: if op == VaeOperation::Encode {
                        1. / 16.
                    } else {
                        16.
                    },
                    channels: if op == VaeOperation::Encode {
                        self.latent_channels
                    } else {
                        self.output_channels
                    },
                    output_device: self.output_device,
                    default_kind: self.default_kind,
                },
                |x| self.run(op, x, services, true),
            )
        });
        let mut result = outputs.next().unwrap()?;
        for next in outputs {
            let next = next?;
            if op == VaeOperation::Encode {
                let _ = result.f_add_(&next)?;
            } else {
                result = result.f_add(&next)?;
            }
        }
        if op == VaeOperation::Encode {
            let _ = result.f_div_scalar_(3.)?;
        } else {
            result = self.process_output(result.f_div_scalar(3.)?)?;
        }
        Ok(result)
    }
}
