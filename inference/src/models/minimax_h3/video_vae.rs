//! Source H3 causal encoder, ViT3D decoder and bounded spatial/temporal IO.
//! Parameters retain their prepared owners; free-memory and attention policy
//! are supplied by the execution owner rather than guessed by the model.
use crate::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21::{Linear, TextEncoderExecution},
    native::Error,
    storage::ScalarField,
};
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

#[derive(Deserialize)]
pub struct Config {
    pub space_down: Vec<i64>,
    pub time_down: Vec<i64>,
    pub num_res_blocks: Vec<usize>,
    pub num_layers: usize,
    pub heads: i64,
    pub dim_head: i64,
    pub num_register_tokens: i64,
    pub clip_length: i64,
    pub token_drop: i64,
    pub tile_size: i64,
    pub tile_overlap_min: i64,
    pub tiling: bool,
    pub still_frame: Option<i64>,
    /// Original runtime policy, independent of whether a layer has INT8 weights.
    pub int8_attention_available: bool,
    /// Original opt-in, false for the selected default profile.
    pub fp16_accumulation: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            space_down: vec![2, 2, 2, 2, 1, 1],
            time_down: vec![1, 2, 2, 1, 1, 1],
            num_res_blocks: vec![2; 6],
            num_layers: 36,
            heads: 32,
            dim_head: 64,
            num_register_tokens: 4,
            clip_length: 17,
            token_drop: 3,
            tile_size: 256,
            tile_overlap_min: 64,
            tiling: true,
            still_frame: None,
            int8_attention_available: true,
            fp16_accumulation: false,
        }
    }
}
pub trait VideoExecution: TextEncoderExecution {
    /// Source get_free_memory(device), queried once for each decoded tile row.
    fn free_memory(&mut self, device: Device) -> Result<u64, Error>;
}
pub struct VideoVae {
    parameters: PreparedParameters,
    pub config: Config,
}
fn ints(xs: &[i64]) -> V {
    V::List(xs.iter().copied().map(V::Int).collect())
}
fn slice(x: &Tensor, dim: i64, start: i64, end: i64) -> Result<Tensor, Error> {
    Ok(x.f_slice(dim, start, end, 1)?)
}
impl VideoVae {
    pub fn new(parameters: PreparedParameters, config: Config) -> Self {
        Self { parameters, config }
    }
    fn get(&self, n: &str) -> Result<Tensor, Error> {
        self.parameters
            .get(n)
            .ok_or_else(|| Error::BindingGap {
                backend: "minimax_h3_video_vae".into(),
                operation: format!("Missing {n}"),
            })?
            .as_tensor()
    }
    fn cast(&self, n: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(self
            .get(n)?
            .f_to_device_(x.device(), x.kind(), false, false)?)
    }
    fn ratio(&self) -> i64 {
        self.config.space_down.iter().product()
    }
    fn ratio_t(&self) -> i64 {
        self.config.time_down.iter().product()
    }
    fn pre_padding(&self) -> i64 {
        (-self.config.clip_length).rem_euclid(self.ratio_t())
    }
    fn tokens_chunk(&self) -> i64 {
        (self.config.clip_length + self.ratio_t() - 1) / self.ratio_t()
    }
    fn token_overlap(&self) -> i64 {
        (-self.config.token_drop).rem_euclid(self.tokens_chunk())
    }
    fn frame_overlap(&self) -> i64 {
        (self.token_overlap() * self.ratio_t() - self.pre_padding()).max(0)
    }
    fn fp16(&self, x: &Tensor) -> bool {
        self.config.fp16_accumulation
            && x.kind() == Kind::Half
            && matches!(x.device(), Device::Cuda(_))
    }
    unsafe fn channels_last(rt: &Runtime, x: &Tensor) -> Result<Tensor, Error> {
        unsafe {
            rt.op(
                "aten::contiguous",
                &[V::tensor(x.shallow_clone())],
                &vec![("memory_format".into(), V::Int(3))],
            )?
        }
        .as_tensor()
    }
    fn group_norm(&self, n: &str, x: &Tensor) -> Result<Tensor, Error> {
        let s = x.size();
        let y = x.f_permute([0, 2, 1, 3, 4])?.f_contiguous()?.f_view([
            s[0] * s[2],
            s[1],
            1,
            s[3],
            s[4],
        ])?;
        Ok(y.f_group_norm(
            32,
            Some(&self.cast(&format!("{n}.weight"), &y)?),
            Some(&self.cast(&format!("{n}.bias"), &y)?),
            1e-6,
            true,
        )?
        .f_view([s[0], s[2], s[1], s[3], s[4]])?
        .f_permute([0, 2, 1, 3, 4])?
        .f_contiguous()?)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve causal convolution operands and fused source branches"
    )]
    unsafe fn conv<E: VideoExecution>(
        &self,
        rt: &Runtime,
        n: &str,
        x: &Tensor,
        stride: [i64; 3],
        padding: [i64; 3],
        norm: Option<&str>,
        spatial: Option<[i64; 4]>,
        residual: Option<&Tensor>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let spatial = spatial.unwrap_or([padding[2], padding[2], padding[1], padding[1]]);
        let front = if x.size()[2] == 1 { 0 } else { padding[0] * 2 };
        // Selected wheel is CUDA (not HIP), Kitchen has the fused pad kernel.
        let ndhwc = matches!(x.device(), Device::Cuda(_))
            && matches!(x.kind(), Kind::Half | Kind::BFloat16);
        let mut weight = self.get(&format!("{n}.weight"))?;
        if ndhwc && matches!(weight.device(), Device::Cuda(_)) {
            let cl = unsafe { Self::channels_last(rt, &weight)? };
            weight.f_set_data(&cl)?;
        }
        let mut y = x.shallow_clone();
        if norm.is_some() || front != 0 || spatial.iter().any(|p| *p != 0) {
            if ndhwc && x.size()[1] % 8 == 0 {
                let (w, b) = if let Some(norm) = norm {
                    (
                        V::tensor(self.cast(&format!("{norm}.weight"), x)?),
                        V::tensor(self.cast(&format!("{norm}.bias"), x)?),
                    )
                } else {
                    (V::None, V::None)
                };
                y = unsafe {
                    rt.call_kitchen(
                        "group_norm_silu_pad3d",
                        &vec![
                            ("x".into(), V::tensor(y)),
                            ("weight".into(), w),
                            ("bias".into(), b),
                            (
                                "num_groups".into(),
                                V::Int(if norm.is_some() { 32 } else { 1 }),
                            ),
                            (
                                "eps".into(),
                                V::Float(if norm.is_some() { 1e-6 } else { 0. }),
                            ),
                            (
                                "pad".into(),
                                ints(&[spatial[0], spatial[1], spatial[2], spatial[3], front]),
                            ),
                            ("silu".into(), V::Bool(norm.is_some())),
                        ],
                    )?
                }
                .as_tensor()?;
            } else {
                if let Some(norm) = norm {
                    y = self.group_norm(norm, &y)?.f_silu_()?;
                }
                if spatial.iter().any(|p| *p != 0) {
                    y = y.f_pad(
                        [spatial[0], spatial[1], spatial[2], spatial[3], 0, 0],
                        "reflect",
                        None,
                    )?;
                }
                if front != 0 {
                    y = y.f_pad([0, 0, 0, 0, front, 0], "constant", None)?;
                }
            }
        }
        let mut weight = weight.f_to_device_(y.device(), y.kind(), false, false)?;
        let bias = self
            .parameters
            .get(&format!("{n}.bias"))
            .map(|v| {
                Ok::<_, Error>(
                    v.as_tensor()?
                        .f_to_device_(y.device(), y.kind(), false, false)?,
                )
            })
            .transpose()?;
        if y.size()[2] == 1 && padding[0] != 0 {
            weight = weight.f_slice(2, -y.size()[2], i64::MAX, 1)?;
        } else if ndhwc {
            weight = unsafe { Self::channels_last(rt, &weight)? };
            if self.fp16(&y) {
                return unsafe {
                    rt.call_kitchen(
                        "fp16_conv3d",
                        &vec![
                            ("x".into(), V::tensor(y)),
                            ("weight".into(), V::tensor(weight)),
                            ("bias".into(), bias.map_or(V::None, V::tensor)),
                            (
                                "residual".into(),
                                residual.map_or(V::None, |t| V::tensor(t.shallow_clone())),
                            ),
                            ("stride".into(), ints(&stride)),
                        ],
                    )?
                }
                .as_tensor();
            }
        }
        // Comfy's cuDNN 2.9/2.10 workaround is false for the locked Torch 2.13.
        let mut out = y.f_conv3d(&weight, bias.as_ref(), stride, [0, 0, 0], [1, 1, 1], 1)?;
        if let Some(r) = residual {
            let _ = out.f_add_(r)?;
        }
        Ok(out)
    }
    unsafe fn moments<E: VideoExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let mut h = unsafe {
            self.conv(
                rt,
                "encoder.conv_in",
                x,
                [1; 3],
                [1; 3],
                None,
                None,
                None,
                exec,
            )?
        };
        for (level, blocks) in self.config.num_res_blocks.iter().enumerate() {
            for block in 0..*blocks {
                let n = format!("encoder.down.{level}.block.{block}");
                if self.parameters.contains_key(&format!("{n}.conv1.weight")) {
                    let y = unsafe {
                        self.conv(
                            rt,
                            &format!("{n}.conv1"),
                            &h,
                            [1; 3],
                            [1; 3],
                            Some(&format!("{n}.norm1")),
                            None,
                            None,
                            exec,
                        )?
                    };
                    let residual = if self
                        .parameters
                        .contains_key(&format!("{n}.nin_shortcut.weight"))
                    {
                        unsafe {
                            self.conv(
                                rt,
                                &format!("{n}.nin_shortcut"),
                                &h,
                                [1; 3],
                                [0; 3],
                                None,
                                None,
                                None,
                                exec,
                            )?
                        }
                    } else {
                        h
                    };
                    h = unsafe {
                        self.conv(
                            rt,
                            &format!("{n}.conv2"),
                            &y,
                            [1; 3],
                            [1; 3],
                            Some(&format!("{n}.norm2")),
                            None,
                            Some(&residual),
                            exec,
                        )?
                    };
                } else if self
                    .parameters
                    .contains_key(&format!("{n}.nin_shortcut.weight"))
                {
                    h = unsafe {
                        self.conv(
                            rt,
                            &format!("{n}.nin_shortcut"),
                            &h,
                            [1; 3],
                            [0; 3],
                            None,
                            None,
                            None,
                            exec,
                        )?
                    };
                }
            }
            let s = self.config.space_down[level];
            let t = self.config.time_down[level];
            if s * t > 1 {
                h = unsafe {
                    self.conv(
                        rt,
                        &format!("encoder.down.{level}.downsample.conv"),
                        &h,
                        [t, s, s],
                        [1, 0, 0],
                        None,
                        (s == 2).then_some([0, 1, 0, 1]),
                        None,
                        exec,
                    )?
                };
            }
        }
        h = unsafe {
            self.conv(
                rt,
                "encoder.conv_out",
                &h,
                [1; 3],
                [1; 3],
                Some("encoder.norm_out"),
                None,
                None,
                exec,
            )?
        };
        unsafe { self.conv(rt, "quant_conv", &h, [1; 3], [0; 3], None, None, None, exec) }
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Source linear_input_act activation and residual operands"
    )]
    unsafe fn linear<E: VideoExecution>(
        &self,
        rt: &Runtime,
        n: &str,
        x: &Tensor,
        act: Option<&str>,
        norm: Option<&str>,
        residual: Option<(&Tensor, &Tensor)>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let layer = Linear::from_parameters(
            &self.parameters,
            n,
            self.parameters.contains_key(&format!("{n}.bias")),
        )?;
        let norm = norm
            .map(|n| self.cast(&format!("{n}.weight"), x))
            .transpose()?;
        if !layer.full_precision
            && !layer.force_forward
            && layer.weight_patches.is_empty()
            && layer.bias_patches.is_empty()
            && let Ok(q) = layer.weight.as_quantized()
            && q.layout == "TensorWiseINT8Layout"
            && q.params.fields.get("transposed") != Some(&ScalarField::Bool(true))
        {
            let w = unsafe {
                rt.op(
                    "aten::to.dtype_layout",
                    std::slice::from_ref(&layer.weight),
                    &vec![
                        ("dtype".into(), V::DType(x.kind())),
                        ("device".into(), V::Device(x.device())),
                    ],
                )?
            }
            .as_quantized()?;
            let group = match w.params.fields.get("convrot_groupsize") {
                Some(ScalarField::Integer(g)) => *g,
                _ => 256,
            };
            let b = layer
                .bias
                .as_ref()
                .map(|b| {
                    Ok::<_, Error>(V::tensor(b.as_tensor()?.f_to_device_(
                        x.device(),
                        x.kind(),
                        false,
                        false,
                    )?))
                })
                .transpose()?
                .unwrap_or(V::None);
            return unsafe {
                rt.call_kitchen(
                    "int8_linear",
                    &vec![
                        ("x".into(), V::tensor(x.shallow_clone())),
                        ("weight".into(), V::tensor(w.payload.shallow_clone())),
                        (
                            "weight_scale".into(),
                            V::tensor(w.params.tensors["scale"].shallow_clone()),
                        ),
                        ("bias".into(), b),
                        ("out_dtype".into(), V::DType(x.kind())),
                        (
                            "convrot".into(),
                            V::Bool(
                                w.params.fields.get("convrot") == Some(&ScalarField::Bool(true)),
                            ),
                        ),
                        ("convrot_groupsize".into(), V::Int(group)),
                        (
                            "input_act".into(),
                            act.map_or(V::None, |a| V::Text(a.into())),
                        ),
                        (
                            "input_act_weight".into(),
                            norm.as_ref()
                                .map_or(V::None, |t| V::tensor(t.shallow_clone())),
                        ),
                        (
                            "input_act_eps".into(),
                            V::Float(if norm.is_some() { 1e-5 } else { 0. }),
                        ),
                        (
                            "residual".into(),
                            residual.map_or(V::None, |(r, _)| V::tensor(r.shallow_clone())),
                        ),
                        (
                            "residual_scale".into(),
                            residual.map_or(V::None, |(_, s)| V::tensor(s.shallow_clone())),
                        ),
                    ],
                )?
            }
            .as_tensor();
        }
        let x = match act {
            Some("rms_norm") => {
                x.f_rms_norm([*x.size().last().unwrap()], norm.as_ref(), Some(1e-5))?
            }
            Some("swiglu") => {
                let p = x.f_chunk(2, -1)?;
                p[0].f_silu()?.f_mul_(&p[1])?
            }
            None => x.shallow_clone(),
            Some(a) => {
                return Err(Error::BindingGap {
                    backend: "minimax_h3_video_vae".into(),
                    operation: format!("activation {a}"),
                });
            }
        };
        if self.fp16(&x)
            && (act.is_some() || residual.is_some())
            && layer.weight.as_quantized().is_err()
            && !layer.force_forward
            && !layer.full_precision
        {
            let w = self.cast(&format!("{n}.weight"), &x)?;
            let b = layer
                .bias
                .as_ref()
                .map(|b| {
                    Ok::<_, Error>(V::tensor(b.as_tensor()?.f_to_device_(
                        x.device(),
                        x.kind(),
                        false,
                        false,
                    )?))
                })
                .transpose()?
                .unwrap_or(V::None);
            return unsafe {
                rt.call_kitchen(
                    "fp16_linear",
                    &vec![
                        ("x".into(), V::tensor(x)),
                        ("weight".into(), V::tensor(w)),
                        ("bias".into(), b),
                        (
                            "residual".into(),
                            residual.map_or(V::None, |(r, _)| V::tensor(r.shallow_clone())),
                        ),
                        (
                            "residual_scale".into(),
                            residual.map_or(V::None, |(_, s)| V::tensor(s.shallow_clone())),
                        ),
                    ],
                )?
            }
            .as_tensor();
        }
        let out = unsafe { layer.forward(rt, &x)? };
        Ok(if let Some((r, s)) = residual {
            r.f_addcmul(&out, s)?
        } else {
            out
        })
    }
    fn residual_scale(&self, n: &str, x: &Tensor) -> Result<Tensor, Error> {
        let s = self.get(n)?;
        if s.device() == x.device() && s.kind() == x.kind() {
            Ok(s)
        } else {
            Ok(s.f_to_device_(x.device(), x.kind(), false, true)?)
        }
    }
    fn rope(&self, z: &Tensor) -> Result<Tensor, Error> {
        let s = z.size();
        let mut coords = vec![];
        for dim in &s[2..] {
            coords.push(
                Tensor::f_arange_start(0.5, *dim as f64, (z.kind(), z.device()))?
                    .f_div_scalar(*dim)?
                    .f_mul_scalar(2.)?
                    .f_sub_scalar(1.)?,
            );
        }
        let grid = Tensor::f_meshgrid_indexing(&coords, "ij")?;
        let ids = Tensor::f_stack(&grid, -1)?
            .f_flatten(0, 2)?
            .f_unsqueeze(0)?
            .f_expand([s[0], -1, -1], false)?;
        let ids = Tensor::f_cat(
            &[
                ids,
                Tensor::f_zeros(
                    [s[0], 1 + self.config.num_register_tokens, 3],
                    (z.kind(), z.device()),
                )?,
            ],
            1,
        )?;
        let freq = self
            .get("decoder.pos_embed.inv_freq")?
            .f_to_device(z.device())?
            .f_view([1, 1, 1, -1])?;
        let angle = ids
            .f_unsqueeze(-1)?
            .f_to_kind(Kind::Float)?
            .f_mul_scalar(2. * std::f64::consts::PI)?
            .f_mul(&freq)?
            .f_flatten(2, 3)?;
        let c = angle.f_cos()?;
        let sin = angle.f_sin()?;
        let shape = angle.size();
        Ok(
            Tensor::f_stack(&[c.shallow_clone(), sin.f_neg()?, sin, c], -1)?
                .f_reshape([shape[0], shape[1], 1, shape[2], 2, 2])?
                .f_to_kind(z.kind())?,
        )
    }
    unsafe fn pixels<E: VideoExecution>(
        &self,
        rt: &Runtime,
        z: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let x = unsafe {
            self.conv(
                rt,
                "post_quant_conv",
                z,
                [1; 3],
                [0; 3],
                None,
                None,
                None,
                exec,
            )?
        };
        let s = x.size();
        let b = s[0];
        let patches = s[2] * s[3] * s[4];
        let c = &self.config;
        let mut h = unsafe {
            self.linear(
                rt,
                "decoder.x_embedder",
                &x.f_flatten(2, -1)?.f_transpose(1, 2)?,
                None,
                None,
                None,
                exec,
            )?
        };
        let registers = self
            .get("decoder.register_tokens")?
            .f_to_device_(h.device(), h.kind(), false, true)?
            .f_expand([b, -1, -1], false)?;
        let zeros = h.f_slice(1, 0, 1, 1)?.f_zeros_like()?;
        h = Tensor::f_cat(&[h, registers, zeros], 1)?;
        let rope = self.rope(&x)?;
        exec.queue_begin(c.num_layers, h.device())?;
        for i in 0..c.num_layers {
            exec.block(i, &mut |exec| {
                let n = format!("decoder.transformer_blocks.{i}");
                let qkv = unsafe {
                    self.linear(
                        rt,
                        &format!("{n}.attn.to_qkv"),
                        &h,
                        Some("rms_norm"),
                        Some(&format!("{n}.norm1")),
                        None,
                        exec,
                    )?
                }
                .f_view([b, -1, c.heads, 3 * c.dim_head])?;
                let p = qkv.f_chunk(3, -1)?;
                let scale = self
                    .get(&format!("{n}.attn.qk_norm_scale"))?
                    .f_to_device(h.device())?;
                let _ = unsafe {
                    rt.call_kitchen(
                        "rms_rope_split_half_",
                        &vec![
                            ("q".into(), V::tensor(p[0].shallow_clone())),
                            ("k".into(), V::tensor(p[1].shallow_clone())),
                            ("freqs_cis".into(), V::tensor(rope.shallow_clone())),
                            ("q_scale".into(), V::tensor(scale)),
                            ("epsilon".into(), V::Float(1e-5)),
                            ("rot_dim".into(), V::Int(rope.size()[3] * 2)),
                        ],
                    )?
                };
                let q = p[0].f_transpose(1, 2)?;
                let k = p[1].f_transpose(1, 2)?;
                let v = p[2].f_transpose(1, 2)?;
                let quantized = self
                    .parameters
                    .get(&format!("{n}.attn.to_qkv.weight"))
                    .unwrap()
                    .as_quantized()
                    .is_ok();
                let out = if quantized && c.int8_attention_available {
                    unsafe { rt.int8_attention(&q, &k, &v, None, None)? }
                        .f_transpose(1, 2)?
                        .f_reshape([b, -1, c.heads * c.dim_head])?
                } else {
                    exec.attention(rt, &q, &k, &v, None, c.heads, false)?
                };
                let scale = self.residual_scale(&format!("{n}.scale1"), &h)?;
                h = unsafe {
                    self.linear(
                        rt,
                        &format!("{n}.attn.to_out"),
                        &out.f_nan_to_num(None, None, None)?,
                        None,
                        None,
                        Some((&h, &scale)),
                        exec,
                    )?
                };
                let f = unsafe {
                    self.linear(
                        rt,
                        &format!("{n}.ff.w1"),
                        &h,
                        Some("rms_norm"),
                        Some(&format!("{n}.norm2")),
                        None,
                        exec,
                    )?
                };
                let scale = self.residual_scale(&format!("{n}.scale2"), &h)?;
                h = unsafe {
                    self.linear(
                        rt,
                        &format!("{n}.ff.w2"),
                        &f,
                        Some("swiglu"),
                        None,
                        Some((&h, &scale)),
                        exec,
                    )?
                };
                exec.observe(&format!("video.decoder.block.{i}"), &h)?;
                Ok(())
            })?;
        }
        exec.queue_end(h.device())?;
        let w = self.cast("decoder.norm_out.weight", &h)?;
        let bias = self.cast("decoder.norm_out.bias", &h)?;
        let h = h.f_layer_norm(w.size().as_slice(), Some(&w), Some(&bias), 1e-5, true)?;
        let out = unsafe { self.linear(rt, "decoder.proj_out", &h, None, None, None, exec)? }
            .f_slice(1, 0, patches, 1)?;
        let r = self.ratio();
        let t = self.ratio_t();
        Ok(out
            .f_view([b, s[2], s[3], s[4], 3, t, r, r].as_slice())?
            .f_permute([0, 4, 1, 5, 2, 6, 3, 7])?
            .f_contiguous()?
            .f_reshape([b, 3, s[2] * t, s[3] * r, s[4] * r])?)
    }
    fn normalize(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_add_scalar(1.)?
            .f_mul_scalar_(0.5)?
            .f_sub_(&self.cast("pixel_mean", x)?)?
            .f_div_(&self.cast("pixel_std", x)?)?)
    }
    fn finalize(&self, x: &Tensor) -> Result<Tensor, Error> {
        let std = self
            .get("pixel_std")?
            .f_to_device_(x.device(), Kind::Float, false, false)?;
        let mean = self
            .get("pixel_mean")?
            .f_to_device_(x.device(), Kind::Float, false, false)?;
        Ok(x.f_mul(&std)?.f_add_(&mean)?.f_clamp_(0., 1.)?)
    }
    pub fn split_tiles(&self, len: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let tile = self.config.tile_size;
        if tile >= len {
            return (vec![0], vec![len], vec![]);
        }
        let mut n = (len + tile - 1) / tile;
        let (mut overlap, remaining) = loop {
            let overlap = vec![self.config.tile_overlap_min; (n - 1) as usize];
            let remaining = tile * n - overlap.iter().sum::<i64>() - len;
            if remaining >= 0 {
                break (overlap, remaining);
            }
            n += 1;
        };
        for i in 0..remaining / self.ratio() {
            overlap[(i % (n - 1)) as usize] += self.ratio();
        }
        let mut starts = vec![0];
        for o in &overlap {
            starts.push(starts.last().unwrap() + tile - o);
        }
        (starts, vec![tile; n as usize], overlap)
    }
    pub fn blend(a: &Tensor, b: &Tensor, extent: i64, dim: i64) -> Result<Tensor, Error> {
        let axis = dim.rem_euclid(a.dim() as i64) as usize;
        let extent = extent.min(a.size()[axis]).min(b.size()[axis]);
        let positions = Tensor::f_arange(extent, (b.kind(), b.device()))?;
        let wa = positions.f_div_scalar(extent)?.f_neg()?.f_add_scalar(1.)?;
        let wb = positions.f_div_scalar(extent)?;
        let mut shape = vec![1; a.dim()];
        shape[axis] = extent;
        // Python -0 is 0; retain that observable source slicing boundary.
        let start = if extent == 0 {
            0
        } else {
            a.size()[axis] - extent
        };
        let blended = slice(a, axis as i64, start, i64::MAX)?
            .f_mul(&wa.f_view(shape.as_slice())?)?
            .f_add(&slice(b, axis as i64, 0, extent)?.f_mul(&wb.f_view(shape.as_slice())?)?)?;
        Ok(if extent < b.size()[axis] {
            Tensor::f_cat(
                &[blended, slice(b, axis as i64, extent, i64::MAX)?],
                axis as i64,
            )?
        } else {
            blended
        })
    }
    unsafe fn adaptive_encode<E: VideoExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        if !self.config.tiling {
            return unsafe { self.moments(rt, x, exec) };
        }
        let s = x.size();
        let (y, yl, yo) = self.split_tiles(s[3]);
        let (xx, xl, xo) = self.split_tiles(s[4]);
        let mut rows = vec![];
        for (i, l) in y.iter().zip(&yl) {
            let mut row = vec![];
            for (j, w) in xx.iter().zip(&xl) {
                row.push(unsafe {
                    self.moments(rt, &slice(&slice(x, 3, *i, i + l)?, 4, *j, j + w)?, exec)?
                });
            }
            rows.push(row);
        }
        let mut result = vec![];
        for (i, row) in rows.iter().enumerate() {
            let mut parts = vec![];
            for (j, tile) in row.iter().enumerate() {
                let mut tile = tile.shallow_clone();
                if i > 0 {
                    tile = Self::blend(&rows[i - 1][j], &tile, yo[i - 1] / self.ratio(), 3)?;
                }
                if j > 0 {
                    tile = Self::blend(&row[j - 1], &tile, xo[j - 1] / self.ratio(), 4)?;
                }
                if i < rows.len() - 1 {
                    tile = slice(&tile, 3, 0, -(yo[i] / self.ratio()))?;
                }
                if j < row.len() - 1 {
                    tile = slice(&tile, 4, 0, -(xo[j] / self.ratio()))?;
                }
                parts.push(tile);
            }
            result.push(Tensor::f_cat(&parts, 4)?);
        }
        Ok(Tensor::f_cat(&result, 3)?)
    }
    unsafe fn adaptive_decode<E: VideoExecution>(
        &self,
        rt: &Runtime,
        z: &Tensor,
        frames: Option<(i64, i64)>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let select = |x: &Tensor| -> Result<Tensor, Error> {
            if let Some((a, b)) = frames {
                slice(x, 2, a, b)
            } else {
                Ok(x.shallow_clone())
            }
        };
        if !self.config.tiling {
            return select(&unsafe { self.pixels(rt, z, exec)? });
        }
        let r = self.ratio();
        let height = z.size()[3] * r;
        let width = z.size()[4] * r;
        let (y, yl, yo) = self.split_tiles(height);
        let (x, xl, xo) = self.split_tiles(width);
        let mut canvas: Option<Tensor> = None;
        let mut strip: Option<Tensor> = None;
        let mut out_y = 0;
        for (i, (pos, len)) in y.iter().zip(&yl).enumerate() {
            let row = slice(z, 3, pos / r, (pos + len) / r)?;
            let free = exec.free_memory(row.device())?;
            let batch = (free / (128 * 2u64.pow(20) * row.size()[0] as u64)).clamp(1, 4) as usize;
            let mut new_strip: Option<Tensor> = None;
            let mut left: Option<Tensor> = None;
            let mut out_x = 0;
            let mut tile_height = 0;
            // Own one decoder batch at a time; never collect a whole decoded row.
            for start in (0..x.len()).step_by(batch) {
                let end = (start + batch).min(x.len());
                let group = (start..end)
                    .map(|j| slice(&row, 4, x[j] / r, (x[j] + xl[j]) / r))
                    .collect::<Result<Vec<_>, _>>()?;
                let decoded = unsafe { self.pixels(rt, &Tensor::f_cat(&group, 0)?, exec)? };
                for (j, raw) in (start..end).zip(decoded.f_chunk((end - start) as i64, 0)?) {
                    let mut tile = select(&raw)?;
                    if i > 0 {
                        tile = Self::blend(
                            &slice(strip.as_ref().unwrap(), 4, x[j], x[j] + xl[j])?,
                            &tile,
                            yo[i - 1],
                            3,
                        )?;
                    }
                    if j > 0 {
                        tile = Self::blend(left.as_ref().unwrap(), &tile, xo[j - 1], 4)?;
                    }
                    left = if j < x.len() - 1 {
                        Some(
                            unsafe {
                                rt.op(
                                    "aten::clone",
                                    &[V::tensor(slice(&tile, 4, -xo[j], i64::MAX)?)],
                                    &vec![],
                                )?
                            }
                            .as_tensor()?,
                        )
                    } else {
                        None
                    };
                    if j < x.len() - 1 {
                        tile = slice(&tile, 4, 0, -xo[j])?;
                    }
                    if canvas.is_none() {
                        let mut shape = tile.size();
                        shape[3] = height;
                        shape[4] = width;
                        canvas = Some(Tensor::f_empty(
                            shape.as_slice(),
                            (tile.kind(), tile.device()),
                        )?);
                    }
                    if i < y.len() - 1 {
                        if new_strip.is_none() {
                            let mut shape = tile.size();
                            shape[3] = yo[i];
                            shape[4] = width;
                            new_strip = Some(Tensor::f_empty(
                                shape.as_slice(),
                                (tile.kind(), tile.device()),
                            )?);
                        }
                        slice(
                            new_strip.as_ref().unwrap(),
                            4,
                            out_x,
                            out_x + tile.size()[4],
                        )?
                        .f_copy_(&slice(&tile, 3, -yo[i], i64::MAX)?)?;
                        tile = slice(&tile, 3, 0, -yo[i])?;
                    }
                    slice(
                        &slice(canvas.as_ref().unwrap(), 3, out_y, out_y + tile.size()[3])?,
                        4,
                        out_x,
                        out_x + tile.size()[4],
                    )?
                    .f_copy_(&tile)?;
                    tile_height = tile.size()[3];
                    out_x += tile.size()[4];
                }
            }
            strip = new_strip;
            out_y += tile_height;
        }
        canvas.ok_or_else(|| Error::Native("Empty H3 tile canvas".into()))
    }
    pub fn temporal_chunks(&self, len: i64) -> (i64, i64) {
        let chunk = self.tokens_chunk();
        let pseudo = len + self.config.token_drop;
        let mut pad = (-pseudo).rem_euclid(chunk);
        let mut count = (pseudo + pad) / chunk - i64::from(self.config.token_drop > 0);
        if count < 1 {
            pad += chunk;
            count += 1;
        }
        (pad, count)
    }
    fn pad_frames(&self, len: i64, pad: i64) -> i64 {
        if pad <= 0 {
            return 0;
        }
        let tail = self.config.clip_length % self.ratio_t();
        if tail == 0 {
            return pad * self.ratio_t();
        }
        (0..pad)
            .map(|k| {
                if (len - pad + k) % self.tokens_chunk() == 0 {
                    tail
                } else {
                    self.ratio_t()
                }
            })
            .sum()
    }
    pub fn decode_output_shape(&self, s: &[i64]) -> Vec<i64> {
        let frames = if s[2] == 1 {
            1
        } else {
            let (pad, count) = self.temporal_chunks(s[2]);
            let len = s[2] + pad;
            let chunk = self.tokens_chunk() * self.ratio_t();
            let mut frames = 0;
            let mut overlap = 0;
            for i in 0..count {
                let start = i * self.tokens_chunk();
                let end = start + self.tokens_chunk() + self.token_overlap();
                let clip = (end.min(len) - start.min(len)).max(0) * self.ratio_t();
                for j in 0..(i64::from(self.config.token_drop > 0) + 1) {
                    let a = j * chunk;
                    let b = (a + chunk).min(clip);
                    let f = (b - a - self.pre_padding()).max(0);
                    if j == 0 {
                        frames += f;
                    } else {
                        overlap = f;
                    }
                }
            }
            frames + overlap - self.pad_frames(len, pad)
        };
        vec![s[0], 3, frames, s[3] * self.ratio(), s[4] * self.ratio()]
    }
    /// # Safety
    /// Prepared weights and execution callbacks satisfy the locked native ABI.
    pub unsafe fn encode<E: VideoExecution>(
        &self,
        rt: &Runtime,
        input: &Tensor,
        device: Device,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let x = if input.dim() == 4 {
            input.f_unsqueeze(2)?
        } else {
            input.shallow_clone()
        };
        let moments = if x.size()[2] == 1 {
            let m = unsafe {
                self.adaptive_encode(rt, &self.normalize(&x.f_to_device(device)?)?, exec)?
            };
            slice(&m, 2, -1, i64::MAX)?
        } else {
            let mut parts = vec![];
            let clip = self.config.clip_length;
            for i in 0..(x.size()[2] + clip - 1) / clip {
                let mut part = slice(&x, 2, i * clip, (i + 1) * clip)?.f_to_device(device)?;
                if part.size()[2] < clip {
                    let pad = slice(&part, 2, -1, i64::MAX)?.f_repeat([
                        1,
                        1,
                        clip - part.size()[2],
                        1,
                        1,
                    ])?;
                    part = Tensor::f_cat(&[part, pad], 2)?;
                }
                parts.push(unsafe { self.adaptive_encode(rt, &self.normalize(&part)?, exec)? });
            }
            let moments = Tensor::f_cat(&parts, 2)?;
            if self.config.token_drop > 0 {
                slice(&moments, 2, 0, -self.config.token_drop)?
            } else {
                moments
            }
        };
        let mean = moments.f_to_kind(Kind::Float)?.f_chunk(2, 1)?.remove(0);
        let lm = self.cast("latents_mean", &mean)?.f_view([1, -1, 1, 1, 1])?;
        let ls = self.cast("latents_std", &mean)?.f_view([1, -1, 1, 1, 1])?;
        Ok(mean.f_sub(&lm)?.f_div(&ls)?)
    }
    fn write_part(&self, out: &Tensor, part: &Tensor, pos: &mut i64) -> Result<(), Error> {
        if part.size()[2] <= 0 {
            return Ok(());
        }
        let part = self.finalize(part)?;
        let count = part.size()[2].min((out.size()[2] - *pos).max(0));
        if count > 0 {
            slice(out, 2, *pos, *pos + count)?.f_copy_(&slice(&part, 2, 0, count)?)?;
            *pos += count;
        }
        Ok(())
    }
    /// # Safety
    /// Same contracts as encode. An optional output buffer has the source shape.
    pub unsafe fn decode<E: VideoExecution>(
        &self,
        rt: &Runtime,
        input: &Tensor,
        output: Option<&Tensor>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let lm = self.cast("latents_mean", input)?.f_view([1, -1, 1, 1, 1])?;
        let ls = self.cast("latents_std", input)?.f_view([1, -1, 1, 1, 1])?;
        let mut z = input.f_mul(&ls)?.f_add(&lm)?;
        if z.size()[2] == 1 {
            let (z, first) = if let Some(f) = self.config.still_frame {
                (z, f)
            } else {
                (
                    z.f_repeat([1, 1, self.tokens_chunk(), 1, 1])?,
                    self.pre_padding(),
                )
            };
            let out = self.finalize(&unsafe {
                self.adaptive_decode(rt, &z, Some((first, first + 1)), exec)?
            })?;
            if let Some(buffer) = output {
                buffer.shallow_clone().f_copy_(&out)?;
                return Ok(buffer.shallow_clone());
            }
            return Ok(out);
        }
        let out = if let Some(out) = output {
            out.shallow_clone()
        } else {
            Tensor::f_empty(
                self.decode_output_shape(&input.size()),
                (Kind::Float, Device::Cpu),
            )?
        };
        let (pad, count) = self.temporal_chunks(z.size()[2]);
        if pad > 0 {
            let last = slice(&z, 2, -1, i64::MAX)?.f_repeat([1, 1, pad, 1, 1])?;
            z = Tensor::f_cat(&[z, last], 2)?;
        }
        let mut overlap: Option<Tensor> = None;
        let mut pos = 0;
        let chunk = self.tokens_chunk() * self.ratio_t();
        for i in 0..count {
            let start = i * self.tokens_chunk();
            let clip = slice(
                &z,
                2,
                start,
                start + self.tokens_chunk() + self.token_overlap(),
            )?;
            let decoded = unsafe { self.adaptive_decode(rt, &clip, None, exec)? };
            for j in 0..i64::from(self.config.token_drop > 0) + 1 {
                let part = slice(&decoded, 2, j * chunk, (j + 1) * chunk)?;
                let mut part = slice(&part, 2, self.pre_padding(), i64::MAX)?;
                if j == 0 {
                    if let Some(prev) = overlap.take() {
                        part = Self::blend(&prev, &part, self.frame_overlap(), 2)?;
                    }
                    self.write_part(&out, &part, &mut pos)?;
                } else {
                    overlap = Some(part.f_contiguous()?);
                }
            }
            if i == count - 1
                && let Some(part) = overlap.take()
            {
                self.write_part(&out, &part, &mut pos)?;
            }
        }
        Ok(out)
    }
}
