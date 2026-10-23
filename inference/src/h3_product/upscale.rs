//! Trained H3 LatentResizer3D, matching the installed provider's evaluation path.
use crate::native::Error;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, Error>;
const MEAN: [f32; 24] = [
    0.85809034,
    -0.96065915,
    1.066164,
    -0.50903255,
    -0.2727582,
    -1.3675414,
    -0.2553255,
    -0.26907554,
    -0.5376841,
    -0.04640973,
    0.66573703,
    0.19690128,
    -0.5460608,
    -0.4035342,
    -0.23683025,
    0.25928453,
    -0.30133945,
    0.21134199,
    -1.1206849,
    0.35819334,
    -0.042251438,
    0.260483,
    0.22864093,
    0.7056032,
];
const STD: [f32; 24] = [
    1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435, 0.9653138, 1.0569886,
    0.8419489, 0.7729953, 1.8955938, 0.94684184, 0.79968095, 0.449889, 0.71974, 0.6936293,
    2.961095, 2.76942, 3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
];
#[derive(Clone)]
enum Block {
    Res(String),
    Temporal(String),
}
pub struct Upscaler {
    weights: BTreeMap<String, Tensor>,
    input: Vec<Block>,
    output: Vec<Block>,
    overlap: i64,
}
fn error(message: impl Into<String>) -> Error {
    Error::Native(message.into())
}
pub fn canvas(width: u32, height: u32, factor: f64) -> (u32, u32) {
    let align = |v: u32| ((f64::from(v) * factor / 32.).round_ties_even() * 32.).max(16.) as u32;
    (align(width), align(height))
}
impl Upscaler {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = Tensor::read_safetensors(path)?;
        let prefixed = raw.iter().any(|(name, _)| name.starts_with("upscaler."));
        let weights = raw
            .into_iter()
            .filter_map(|(name, t)| {
                if prefixed {
                    name.strip_prefix("upscaler.")
                        .map(|name| (name.to_owned(), t))
                } else {
                    Some((name, t))
                }
            })
            .map(|(name, t)| Ok((name, t.f_to_kind(Kind::Float)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Self::from_weights(weights)
    }
    fn from_weights(weights: BTreeMap<String, Tensor>) -> Result<Self> {
        let temporal = weights.keys().any(|key| key.ends_with("dwconv.weight"));
        let mut expected = BTreeSet::new();
        let mut pair = |prefix: String| {
            expected.insert(format!("{prefix}.weight"));
            expected.insert(format!("{prefix}.bias"));
        };
        for p in ["conv_in", "conv_out", "embed.0", "embed.2", "norm_out"] {
            pair(p.into());
        }
        let mut groups = vec![];
        for name in ["in_blocks", "out_blocks"] {
            let count = weights
                .keys()
                .filter(|key| {
                    key.starts_with(&format!("{name}.")) && key.ends_with("in_layers.0.weight")
                })
                .count();
            let mut blocks = vec![];
            let mut index = 0;
            for i in 0..count {
                let p = format!("{name}.{index}");
                for suffix in [
                    "in_layers.0",
                    "in_layers.2",
                    "emb_layers.1",
                    "out_norm",
                    "out_layers.2",
                ] {
                    pair(format!("{p}.{suffix}"));
                }
                blocks.push(Block::Res(p));
                index += 1;
                if temporal && i % 2 == 0 {
                    let p = format!("{name}.{index}");
                    for suffix in ["norm", "dwconv", "pwconv"] {
                        pair(format!("{p}.{suffix}"));
                    }
                    blocks.push(Block::Temporal(p));
                    index += 1;
                }
            }
            groups.push(blocks);
        }
        if expected != weights.keys().cloned().collect() {
            return Err(error(
                "LatentResizer3D state dict has missing or unexpected parameters",
            ));
        }
        let overlap = weights
            .iter()
            .find(|(key, _)| key.starts_with("in_blocks.") && key.ends_with("dwconv.weight"))
            .map_or(0, |(_, w)| w.size()[2]);
        let output = groups.pop().unwrap();
        let input = groups.pop().unwrap();
        Ok(Self {
            weights,
            input,
            output,
            overlap,
        })
    }
    pub fn to(&mut self, device: Device, kind: Kind) -> Result<()> {
        // The provider moves its FP32 module first, then converts its precision.
        for tensor in self.weights.values_mut() {
            *tensor = tensor.f_to_device(device)?;
        }
        for tensor in self.weights.values_mut() {
            *tensor = tensor.f_to_kind(kind)?;
        }
        Ok(())
    }
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| error(format!("Missing LatentResizer3D parameter {key}")))
    }
    fn linear(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        Ok(x.f_linear(
            self.w(&format!("{p}.weight"))?,
            Some(self.w(&format!("{p}.bias"))?),
        )?)
    }
    fn conv(&self, x: &Tensor, p: &str, groups: i64) -> Result<Tensor> {
        let w = self.w(&format!("{p}.weight"))?;
        let shape = w.size();
        Ok(x.f_conv3d(
            w,
            Some(self.w(&format!("{p}.bias"))?),
            [1, 1, 1],
            [shape[2] / 2, shape[3] / 2, shape[4] / 2],
            [1, 1, 1],
            groups,
        )?)
    }
    fn norm(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        Ok(x.f_group_norm(
            32,
            Some(self.w(&format!("{p}.weight"))?),
            Some(self.w(&format!("{p}.bias"))?),
            1e-5,
            true,
        )?)
    }
    fn block(&self, x: Tensor, emb: &Tensor, block: &Block) -> Result<Tensor> {
        match block {
            Block::Temporal(p) => {
                let h = self.norm(&x, &format!("{p}.norm"))?.f_silu()?;
                let h = self.conv(&h, &format!("{p}.dwconv"), x.size()[1])?;
                Ok(x.f_add(&self.conv(&h, &format!("{p}.pwconv"), 1)?)?)
            }
            Block::Res(p) => {
                let h = self.norm(&x, &format!("{p}.in_layers.0"))?.f_silu()?;
                let h = self.conv(&h, &format!("{p}.in_layers.2"), 1)?;
                let emb = self
                    .linear(
                        &emb.f_expand([x.size()[0], -1], false)?.f_silu()?,
                        &format!("{p}.emb_layers.1"),
                    )?
                    .f_to_kind(h.kind())?
                    .f_unsqueeze(-1)?
                    .f_unsqueeze(-1)?
                    .f_unsqueeze(-1)?;
                let chunks = emb.f_chunk(2, 1)?;
                let h = self
                    .norm(&h, &format!("{p}.out_norm"))?
                    .f_mul(&chunks[0].f_add_scalar(1.)?)?
                    .f_add(&chunks[1])?
                    .f_silu()?;
                Ok(x.f_add(&self.conv(&h, &format!("{p}.out_layers.2"), 1)?)?)
            }
        }
    }
    fn segment(&self, x: &Tensor, factor: f64, size: [i64; 3]) -> Result<Tensor> {
        let emb = Tensor::f_from_slice(&[factor - 1.])?
            .f_to_kind(x.kind())?
            .f_to_device(x.device())?
            .f_reshape([1, 1])?;
        let emb = self.linear(&self.linear(&emb, "embed.0")?.f_silu()?, "embed.2")?;
        let mut out = self.conv(x, "conv_in", 1)?;
        for block in &self.input {
            out = self.block(out, &emb, block)?;
        }
        out = out.f_upsample_trilinear3d(size, false, None, None, None)?;
        for block in &self.output {
            out = self.block(out, &emb, block)?;
        }
        self.conv(&self.norm(&out, "norm_out")?.f_silu()?, "conv_out", 1)
    }
    pub fn forward(
        &self,
        x: &Tensor,
        factor: f64,
        size: [i64; 3],
        chunking: bool,
    ) -> Result<Tensor> {
        if x.size()[2..] == size {
            return Ok(x.shallow_clone());
        }
        let t = x.size()[2];
        if !chunking || t <= 32 {
            return self.segment(x, factor, size);
        }
        let o = self.overlap;
        let padded = x.f_replication_pad3d([0, 0, 0, 0, o, o])?;
        let out = Tensor::f_zeros(
            [x.size()[0], x.size()[1], t, size[1], size[2]],
            (x.kind(), x.device()),
        )?;
        let weights = Tensor::f_zeros([1, 1, t, 1, 1], (x.kind(), x.device()))?;
        for start in (0..t).step_by(32) {
            let end = (start + 32).min(t);
            let out_start = (start - o).max(0);
            let out_end = (end + o).min(t);
            let lo = (out_start - o).max(0);
            let hi = (out_end + o).min(t + 2 * o);
            let segment = padded.f_narrow(2, lo, hi - lo)?.f_contiguous()?;
            let result = self.segment(&segment, factor, [hi - lo, size[1], size[2]])?;
            let valid = result.f_narrow(2, out_start + o - lo, out_end - out_start)?;
            let n = out_end - out_start;
            let weight = Tensor::f_ones([n], (x.kind(), x.device()))?;
            if start > out_start {
                let blend = start - out_start;
                weight.f_narrow(0, 0, blend)?.f_copy_(
                    &(Tensor::f_arange_start(1, blend + 1, (x.kind(), x.device()))?
                        / (blend + 1) as f64),
                )?;
            }
            if out_end > end {
                let blend = out_end - end;
                weight.f_narrow(0, n - blend, blend)?.f_copy_(
                    &(Tensor::f_arange_start_step(blend, 0, -1, (x.kind(), x.device()))?
                        / (blend + 1) as f64),
                )?;
            }
            let weight = weight.f_view([1, 1, n, 1, 1])?;
            let _ = out
                .f_narrow(2, out_start, n)?
                .f_add_(&valid.f_mul(&weight)?)?;
            let _ = weights.f_narrow(2, out_start, n)?.f_add_(&weight)?;
        }
        Ok(out.f_div(&weights.f_clamp_min(1e-8)?)?)
    }
    pub fn video(&mut self, input: &Tensor, factor: f64, device: Device) -> Result<Tensor> {
        let shape = input.size();
        let (w, h) = canvas((shape[4] * 16) as u32, (shape[3] * 16) as u32, factor);
        if (i64::from(w / 16), i64::from(h / 16)) == (shape[4], shape[3]) {
            return Ok(input.shallow_clone());
        }
        self.to(device, Kind::Half)?;
        let result: Result<Tensor> = (|| {
            let source = input.f_to_device_(device, Kind::Half, false, true)?;
            let mean = Tensor::f_from_slice(&MEAN)?
                .f_to_kind(Kind::Half)?
                .f_to_device(device)?
                .f_view([1, 24, 1, 1, 1])?;
            let std = Tensor::f_from_slice(&STD)?
                .f_to_kind(Kind::Half)?
                .f_to_device(device)?
                .f_view([1, 24, 1, 1, 1])?;
            let normalized = source.f_sub(&mean)?.f_div(&std)?;
            let out = self
                .forward(
                    &normalized,
                    factor,
                    [shape[2], i64::from(h / 16), i64::from(w / 16)],
                    true,
                )?
                .f_mul(&std)?
                .f_add(&mean)?;
            Ok(out.f_to_device_(Device::Cpu, input.kind(), true, false)?)
        })();
        // force_unload=True: retain this request's half-precision model on CPU.
        let result = result?;
        for tensor in self.weights.values_mut() {
            *tensor = tensor.f_to_device_(Device::Cpu, tensor.kind(), true, false)?;
        }
        Ok(result)
    }
}
