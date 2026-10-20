//! ComfyUI's default TorchSDE BrownianTree path (halfway tree, tol=1e-6,
//! pool_size=24, cache_size=45, Levy area 'none').
use super::noise::{seed_sequence, words};
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};

struct Node {
    start: f64,
    end: f64,
    parent: Option<usize>,
    left: bool,
    mid: Option<f64>,
    key: Vec<u32>,
    depth: u128,
    seed: u32,
    children: Option<(usize, usize)>,
}
impl Node {
    fn new(start: f64, end: f64, parent: Option<usize>, left: bool) -> Self {
        Self {
            start: round(start),
            end: round(end),
            parent,
            left,
            mid: None,
            key: vec![0],
            depth: 0,
            seed: 0,
            children: None,
        }
    }
}
// Decimal formatting performs exact decimal rounding of the binary value,
// unlike multiply/round/divide (which double-rounds values near halfway ties).
fn round(time: f64) -> f64 {
    format!("{time:.6}").parse().unwrap_or(time)
}
fn multiply(t: &Tensor, value: f64) -> Result<Tensor, Error> {
    Ok(t.f_mul(&Tensor::from(value))?)
}
fn shifted_key(key: &[u32], bit: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut carry = bit;
    for &n in key {
        out.push((n << 1) | carry);
        carry = n >> 31;
    }
    if carry != 0 {
        out.push(carry);
    }
    out
}
struct Tree {
    native: Kitchen,
    entropy: Vec<u32>,
    shape: Vec<i64>,
    kind: Kind,
    device: Device,
    nodes: Vec<Node>,
    root_w: Tensor,
    _root_h: Tensor,
    cache: Vec<(usize, Tensor)>,
}
impl Tree {
    fn new(native: &Kitchen, t0: f64, t1: f64, w0: &Tensor, entropy: u64) -> Result<Self, Error> {
        let entropy = words(entropy as u128);
        let seeds = seed_sequence(&entropy, &[], 24, 3)?;
        let root_w = multiply(
            &native
                .generator(w0.device(), seeds[0] as u64)?
                .randn(&w0.size(), w0.kind())?,
            (t1 - t0).sqrt(),
        )?;
        // TorchSDE retains H even when Levy area computation is disabled.
        let root_h = multiply(
            &native
                .generator(w0.device(), seeds[1] as u64)?
                .randn(&w0.size(), w0.kind())?,
            ((t1 - t0) / 12.).sqrt(),
        )?;
        Ok(Self {
            native: native.clone(),
            entropy,
            shape: w0.size(),
            kind: w0.kind(),
            device: w0.device(),
            nodes: vec![Node::new(t0, t1, None, false)],
            root_w,
            _root_h: root_h,
            cache: Vec::new(),
        })
    }
    fn split(&mut self, id: usize, target: f64) -> Result<(), Error> {
        let mut id = id;
        loop {
            let start = self.nodes[id].start;
            let end = self.nodes[id].end;
            let mid = round(0.5 * (end + start));
            if self.nodes[id].mid.is_none() {
                let (key, depth) = if let Some(parent) = self.nodes[id].parent {
                    (
                        shifted_key(&self.nodes[parent].key, u32::from(!self.nodes[id].left)),
                        self.nodes[parent].depth + 1,
                    )
                } else {
                    (vec![0], 0)
                };
                let seeds = seed_sequence(&self.entropy, &[key.clone(), words(depth)], 24, 4)?;
                let left = self.nodes.len();
                let right = left + 1;
                self.nodes.push(Node::new(start, mid, Some(id), true));
                self.nodes.push(Node::new(mid, end, Some(id), false));
                let node = &mut self.nodes[id];
                node.mid = Some(mid);
                node.key = key;
                node.depth = depth;
                node.seed = seeds[0];
                node.children = Some((left, right));
            }
            if target == mid {
                return Ok(());
            }
            if mid == start || mid == end {
                return Err(Error::ReferenceException {
                    class: "RecursionError",
                    message: "maximum recursion depth exceeded while calling a Python object"
                        .into(),
                });
            }
            let (left, right) = self.nodes[id].children.unwrap();
            id = if target > mid { right } else { left };
        }
    }
    fn locate(&mut self, id: usize, a: f64, b: f64, out: &mut Vec<usize>) -> Result<(), Error> {
        let start = self.nodes[id].start;
        let end = self.nodes[id].end;
        if a == start && b == end {
            out.push(id);
            return Ok(());
        }
        if self.nodes[id].mid.is_none() {
            self.split(id, if a == start { b } else { a })?;
        }
        let mid = self.nodes[id].mid.unwrap();
        let (left, right) = self.nodes[id].children.unwrap();
        if b <= mid {
            self.locate(left, a, b, out)
        } else if a >= mid {
            self.locate(right, a, b, out)
        } else {
            self.locate(left, a, mid, out)?;
            self.locate(right, mid, b, out)
        }
    }
    fn increment(&mut self, id: usize) -> Result<Tensor, Error> {
        if id == 0 {
            return Ok(self.root_w.shallow_clone());
        }
        if let Some((_, value)) = self.cache.iter().find(|(key, _)| *key == id) {
            return Ok(value.shallow_clone());
        }
        let parent = self.nodes[id].parent.unwrap();
        let w = self.increment(parent)?;
        let node = &self.nodes[parent];
        let mid = node.mid.unwrap();
        let reciprocal = 1. / (node.end - node.start);
        let left = mid - node.start;
        let right = node.end - mid;
        let mean = multiply(&multiply(&w, left)?, reciprocal)?;
        let variance = left * right * reciprocal;
        let noise = self
            .native
            .generator(self.device, node.seed as u64)?
            .randn(&self.shape, self.kind)?;
        let left_w = mean.f_add(&multiply(&noise, variance.sqrt())?)?;
        let value = if self.nodes[id].left {
            left_w
        } else {
            w.f_sub(&left_w)?
        };
        // _LRUDict updates insertion order on writes, not on reads.
        if self.cache.len() >= 45 {
            let _ = self.cache.remove(0);
        }
        self.cache.push((id, value.shallow_clone()));
        Ok(value)
    }
    fn query(
        &mut self,
        mut a: f64,
        mut b: f64,
        warning: &mut dyn FnMut(&str),
    ) -> Result<Tensor, Error> {
        let start = self.nodes[0].start;
        let end = self.nodes[0].end;
        let py = crate::registry::python_float;
        if a < start {
            warning(&format!(
                "Should have ta>=t0 but got ta={} and t0={}.",
                py(a),
                py(start)
            ));
            a = start;
        }
        if b < start {
            warning(&format!(
                "Should have tb>=t0 but got tb={} and t0={}.",
                py(b),
                py(start)
            ));
            b = start;
        }
        if a > end {
            warning(&format!(
                "Should have ta<=t1 but got ta={} and t1={}.",
                py(a),
                py(end)
            ));
            a = end;
        }
        if b > end {
            warning(&format!(
                "Should have tb<=t1 but got tb={} and t1={}.",
                py(b),
                py(end)
            ));
            b = end;
        }
        if a > b {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("Query times ta={a:.3} and tb={b:.3} must respect ta <= tb."),
            });
        }
        if a == b {
            return Ok(Tensor::f_zeros(&self.shape, (self.kind, self.device))?);
        }
        let mut intervals = Vec::new();
        self.locate(0, round(a), round(b), &mut intervals)?;
        let mut w = self.increment(intervals[0])?;
        for &id in &intervals[1..] {
            w = w.f_add(&self.increment(id)?)?;
        }
        Ok(w)
    }
}
pub enum Seeds {
    One(u64),
    Batch(Vec<u64>),
    Unseeded,
}
pub type Transform = Box<dyn Fn(&Tensor) -> Result<Tensor, Error>>;
pub struct BrownianNoise {
    trees: Vec<Tree>,
    _w0: Tensor,
    batched: bool,
    cpu: bool,
    sign: f64,
    transform: Transform,
}
impl BrownianNoise {
    pub fn new(
        native: &Kitchen,
        x: &Tensor,
        min: &Tensor,
        max: &Tensor,
        seeds: Seeds,
        cpu: bool,
        transform: Transform,
    ) -> Result<Self, Error> {
        let a = transform(min)?;
        let b = transform(max)?;
        let (a, b, sign) = if a.double_value(&[]) < b.double_value(&[]) {
            (a, b, 1.)
        } else {
            (b, a, -1.)
        };
        let mut w0 = x.f_zeros_like()?;
        let (seeds, batched) = match seeds {
            Seeds::One(seed) => (vec![seed], false),
            Seeds::Batch(seeds) => {
                if seeds.len() as i64 != x.size()[0] {
                    return Err(Error::ReferenceException{class:"ValueError",message:"Passing a list or tuple of seeds to BatchedBrownianTree requires a length matching the batch size.".into()});
                }
                w0 = w0.f_get(0)?;
                (seeds, true)
            }
            Seeds::Unseeded => (
                vec![
                    Tensor::f_randint(i64::MAX, [], (Kind::Int64, Device::Cpu))?.int64_value(&[])
                        as u64,
                ],
                false,
            ),
        };
        if cpu {
            w0 = w0.f_detach()?.f_to_device(Device::Cpu)?;
        }
        let trees = seeds
            .into_iter()
            .map(|seed| Tree::new(native, a.double_value(&[]), b.double_value(&[]), &w0, seed))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            trees,
            _w0: w0,
            batched,
            cpu,
            sign,
            transform,
        })
    }
    pub fn sample(
        &mut self,
        from: &Tensor,
        to: &Tensor,
        warning: &mut dyn FnMut(&str),
    ) -> Result<Tensor, Error> {
        let a = (self.transform)(from)?;
        let b = (self.transform)(to)?;
        let (low, high, sign) = if a.double_value(&[]) < b.double_value(&[]) {
            (&a, &b, 1.)
        } else {
            (&b, &a, -1.)
        };
        let device = low.device();
        let kind = low.kind();
        let (low, high) = if self.cpu {
            (
                low.f_detach()?
                    .f_to_device(Device::Cpu)?
                    .f_to_kind(Kind::Float)?,
                high.f_detach()?
                    .f_to_device(Device::Cpu)?
                    .f_to_kind(Kind::Float)?,
            )
        } else {
            (low.shallow_clone(), high.shallow_clone())
        };
        let values = self
            .trees
            .iter_mut()
            .map(|tree| tree.query(low.double_value(&[]), high.double_value(&[]), warning))
            .collect::<Result<Vec<_>, _>>()?;
        let value = Tensor::f_stack(&values, 0)?
            .f_to_device_(device, kind, false, false)?
            .f_mul_scalar(self.sign * sign)?;
        let value = if self.batched { value } else { value.f_get(0)? };
        Ok(value.f_div(&b.f_sub(&a)?.f_abs()?.f_sqrt()?)?)
    }
    pub fn buffer_bytes(&self) -> Result<u64, Error> {
        let mut storages = std::collections::BTreeMap::new();
        let (id, bytes) = crate::native::storage_info(&self._w0)?;
        storages.insert(id, bytes);
        for tree in &self.trees {
            for tensor in [&tree.root_w, &tree._root_h]
                .into_iter()
                .chain(tree.cache.iter().map(|(_, t)| t))
            {
                let (id, bytes) = crate::native::storage_info(tensor)?;
                storages.insert(id, bytes);
            }
        }
        Ok(storages.values().sum())
    }
}
