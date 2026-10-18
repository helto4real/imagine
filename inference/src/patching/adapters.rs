use super::*;
#[derive(Clone)]
pub struct Adapter {
    pub name: String,
    pub weights: Vec<Value>,
}
fn einsum(eq: &str, ts: &[&Tensor]) -> Result<Tensor, Error> {
    Ok(Tensor::f_einsum(eq, ts, None::<&[i64]>)?)
}
impl Adapter {
    fn t(&self, native: &Kitchen, i: usize, w: &Tensor, k: Kind) -> Result<Tensor, Error> {
        cast(native, &self.weights[i].as_tensor()?, w, k)
    }
    fn has(&self, i: usize) -> bool {
        !matches!(self.weights[i], Value::None)
    }
    fn alpha(&self, i: usize, rank: i64) -> f64 {
        match self.weights[i] {
            Value::Float(a) => a / rank as f64,
            Value::Int(a) => a as f64 / rank as f64,
            _ => 1.,
        }
    }
    pub fn calculate(
        &self,
        native: &Kitchen,
        mut w: Tensor,
        p: &Patch,
        key: &str,
        k: Kind,
        d: &mut Diagnostics,
    ) -> Result<Tensor, Error> {
        let (delta, alpha, dora) = match self.name.as_str() {
            "lora" => {
                let a = self.t(native, 0, &w, k)?;
                let mut b = self.t(native, 1, &w, k)?;
                if let Value::List(shape) = &self.weights[5] {
                    let shape = shape
                        .iter()
                        .map(|x| match x {
                            Value::Int(n) => Ok(*n),
                            _ => Err(gap("Noninteger reshape")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    w = pad_to_shape(&w, &shape)?;
                }
                let alpha = self.alpha(2, b.size()[0]);
                if self.has(3) {
                    let c = self.t(native, 3, &w, k)?;
                    b = b
                        .f_transpose(0, 1)?
                        .f_flatten(1, -1)?
                        .f_mm(&c.f_transpose(0, 1)?.f_flatten(1, -1)?)?
                        .f_reshape([b.size()[1], b.size()[0], c.size()[2], c.size()[3]])?
                        .f_transpose(0, 1)?;
                }
                (
                    a.f_flatten(1, -1)
                        .and_then(|a| a.f_mm(&b.f_flatten(1, -1)?))
                        .and_then(|v| v.f_reshape(w.size())),
                    alpha,
                    4,
                )
            }
            "loha" => {
                let (m1, m2) = if self.has(5) {
                    (
                        einsum(
                            "i j k l, j r, i p -> p r k l",
                            &[
                                &self.t(native, 5, &w, k)?,
                                &self.t(native, 1, &w, k)?,
                                &self.t(native, 0, &w, k)?,
                            ],
                        )?,
                        einsum(
                            "i j k l, j r, i p -> p r k l",
                            &[
                                &self.t(native, 6, &w, k)?,
                                &self.t(native, 4, &w, k)?,
                                &self.t(native, 3, &w, k)?,
                            ],
                        )?,
                    )
                } else {
                    (
                        self.t(native, 0, &w, k)?.f_mm(&self.t(native, 1, &w, k)?)?,
                        self.t(native, 3, &w, k)?.f_mm(&self.t(native, 4, &w, k)?)?,
                    )
                };
                (
                    m1.f_mul(&m2).and_then(|v| v.f_reshape(w.size())),
                    self.alpha(2, self.weights[1].as_tensor()?.size()[0]),
                    7,
                )
            }
            "lokr" => {
                let mut dim = None;
                let mut a = if self.has(0) {
                    self.t(native, 0, &w, k)?
                } else {
                    dim = Some(self.weights[4].as_tensor()?.size()[0]);
                    self.t(native, 3, &w, k)?.f_mm(&self.t(native, 4, &w, k)?)?
                };
                let b = if self.has(1) {
                    self.t(native, 1, &w, k)?
                } else {
                    dim = Some(self.weights[6].as_tensor()?.size()[0]);
                    if self.has(7) {
                        einsum(
                            "i j k l, j r, i p -> p r k l",
                            &[
                                &self.t(native, 7, &w, k)?,
                                &self.t(native, 6, &w, k)?,
                                &self.t(native, 5, &w, k)?,
                            ],
                        )?
                    } else {
                        self.t(native, 5, &w, k)?.f_mm(&self.t(native, 6, &w, k)?)?
                    }
                };
                if b.dim() == 4 {
                    a = a.f_unsqueeze(2)?.f_unsqueeze(2)?;
                }
                (
                    a.f_kron(&b).and_then(|v| v.f_reshape(w.size())),
                    dim.map_or(1., |n| self.alpha(2, n)),
                    8,
                )
            }
            "glora" => {
                let shapes = (0..4)
                    .map(|i| self.weights[i].as_tensor().map(|t| t.size()))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut old = shapes[3][1] == shapes[2][0]
                    && shapes[2][0] == shapes[0][0]
                    && shapes[0][0] == shapes[1][1];
                let mut rank = if old { Some(shapes[0][0]) } else { None };
                if shapes[3][0] == shapes[2][1]
                    && shapes[2][1] == shapes[0][1]
                    && shapes[0][1] == shapes[1][0]
                    && !(old && shapes[1][0] == w.size()[0] && w.size()[0] == w.size()[1])
                {
                    old = false;
                    rank = Some(shapes[1][0]);
                }
                let rank = if self.has(4) {
                    rank.ok_or_else(|| exception("UnboundLocalError", "cannot access local variable 'rank' where it is not associated with a value"))?
                } else {
                    rank.unwrap_or(1)
                };
                let a = self.t(native, 0, &w, k)?.f_flatten(1, -1)?;
                let b = self.t(native, 1, &w, k)?.f_flatten(1, -1)?;
                let c = self.t(native, 2, &w, k)?.f_flatten(1, -1)?;
                let e = self.t(native, 3, &w, k)?.f_flatten(1, -1)?;
                let delta = (|| {
                    if old {
                        return e
                            .f_mm(&c)?
                            .f_add(&w.f_flatten(1, -1)?.f_to_kind(k)?.f_mm(&b)?.f_mm(&a)?)?
                            .f_reshape(w.size());
                    }
                    let mut v = if w.dim() > 2 {
                        einsum(
                            "o i ..., i j -> o j ...",
                            &[
                                &einsum("o i ..., i j -> o j ...", &[&w.f_to_kind(k)?, &a])
                                    .map_err(|e| tch::TchError::Torch(e.to_string()))?,
                                &b,
                            ],
                        )
                        .map_err(|e| tch::TchError::Torch(e.to_string()))?
                        .f_reshape(w.size())?
                    } else {
                        w.f_to_kind(k)?.f_mm(&a)?.f_mm(&b)?.f_reshape(w.size())?
                    };
                    let _ = v.f_add_(&c.f_mm(&e)?.f_reshape(w.size())?)?;
                    Ok(v)
                })();
                (delta, self.alpha(4, rank), 5)
            }
            "oft" | "boft" => {
                let blocks = self.t(native, 0, &w, k)?;
                let rescale = if self.has(1) {
                    Some(self.t(native, 1, &w, k)?)
                } else {
                    None
                };
                let alpha = match self.weights[2] {
                    Value::Float(v) => v,
                    Value::Int(v) => v as f64,
                    _ => 0.,
                };
                let delta = (|| -> Result<Tensor, tch::TchError> {
                    if self.name == "boft" && !self.has(2) {
                        return Err(tch::TchError::Torch(
                            "'>' not supported between instances of 'NoneType' and 'int'".into(),
                        ));
                    }
                    let size = *blocks.size().last().unwrap();
                    let eye = Tensor::f_eye(size, (blocks.kind(), blocks.device()))?;
                    let q = blocks.f_sub(&blocks.f_transpose(-1, -2)?)?;
                    let normed = if alpha > 0. {
                        let norm = q.f_norm()?.f_add_scalar(1e-8)?;
                        if norm.f_double_value(&[])? > alpha {
                            q.f_mul_scalar(alpha)?.f_div(&norm)?
                        } else {
                            q
                        }
                    } else {
                        q
                    };
                    let r = eye
                        .f_add(&normed)?
                        .f_matmul(&eye.f_sub(&normed)?.f_to_kind(Kind::Float)?.f_inverse()?)?
                        .f_to_kind(w.kind())?;
                    if self.name == "oft" {
                        let iw = eye.f_to_kind(w.kind())?;
                        let mut shape = vec![blocks.size()[0], size];
                        shape.extend(&w.size()[1..]);
                        let v = einsum(
                            "k n m, k n ... -> k m ...",
                            &[
                                &r.f_mul_scalar(p.strength)?
                                    .f_sub(&iw.f_mul_scalar(p.strength)?)?,
                                &w.f_view(shape.as_slice())?,
                            ],
                        )
                        .map_err(|e| tch::TchError::Torch(e.to_string()))?;
                        return v.f_reshape(w.size());
                    }
                    let mut inp = w.shallow_clone();
                    for i in 0..blocks.size()[0] {
                        let mut bi = r.f_get(i)?;
                        if p.strength != 1. {
                            bi = bi
                                .f_mul_scalar(p.strength)?
                                .f_add(&eye.f_mul_scalar(1. - p.strength)?)?;
                        }
                        let kk = (1_i64 << i) * (size / 2);
                        let mut shape = vec![-1, 2, kk];
                        shape.extend(&inp.size()[1..]);
                        inp = inp
                            .f_reshape(shape.as_slice())?
                            .f_transpose(1, 2)?
                            .f_flatten(0, 2)?;
                        let mut shape = vec![-1, size];
                        shape.extend(&inp.size()[1..]);
                        inp = inp.f_reshape(shape.as_slice())?;
                        inp = einsum("b i j, b j ...-> b i ...", &[&bi, &inp])
                            .map_err(|e| tch::TchError::Torch(e.to_string()))?;
                        inp = inp.f_flatten(0, 1)?;
                        let mut shape = vec![-1, kk, 2];
                        shape.extend(&inp.size()[1..]);
                        inp = inp
                            .f_reshape(shape.as_slice())?
                            .f_transpose(1, 2)?
                            .f_flatten(0, 2)?;
                    }
                    if let Some(scale) = rescale {
                        inp = inp.f_mul(&scale)?;
                    }
                    inp.f_sub(&w)?.f_to_kind(k)
                })();
                (delta, alpha, 3)
            }
            _ => return Err(gap(format!("Unknown adapter {}", self.name))),
        };
        match delta {
            Err(e) => {
                d.errors.push(format!(
                    "ERROR {} {key} {}",
                    self.name,
                    match &e {
                        tch::TchError::Torch(m) => m
                            .lines()
                            .next()
                            .unwrap_or(m)
                            .trim_start_matches("Internal torch error: ")
                            .to_owned(),
                        _ => e.to_string(),
                    }
                ));
                Ok(w)
            }
            Ok(delta) => {
                let result = (|| -> Result<Tensor, Error> {
                    if self.has(dora) {
                        decompose(
                            native,
                            &self.weights[dora].as_tensor()?,
                            &mut w,
                            delta,
                            alpha,
                            p,
                            k,
                        )?;
                    } else {
                        let a = if self.name == "oft" || self.name == "boft" {
                            p.strength
                        } else {
                            p.strength * alpha
                        };
                        let delta = delta.f_mul_scalar(a)?.f_to_kind(w.kind())?;
                        let _ = w.f_add_(&transform(p, &delta)?)?;
                    }
                    Ok(w.shallow_clone())
                })();
                match result {
                    Ok(w) => Ok(w),
                    Err(e) => {
                        d.errors.push(format!(
                            "ERROR {} {key} {}",
                            self.name,
                            match &e {
                                Error::Torch(tch::TchError::Torch(m)) => m
                                    .lines()
                                    .next()
                                    .unwrap_or(m)
                                    .trim_start_matches("Internal torch error: ")
                                    .to_owned(),
                                _ => e.to_string(),
                            }
                        ));
                        Ok(w)
                    }
                }
            }
        }
    }
}
fn decompose(
    native: &Kitchen,
    scale: &Tensor,
    w: &mut Tensor,
    mut diff: Tensor,
    alpha: f64,
    p: &Patch,
    k: Kind,
) -> Result<(), Error> {
    let scale = cast(native, scale, w, k)?;
    let _ = diff.f_mul_scalar_(alpha)?;
    let mut calc = w.f_add(&transform(p, &diff)?.f_to_kind(w.kind())?)?;
    let norm = if scale.size()[0] == calc.size()[0] {
        let mut shape = vec![w.size()[0]];
        shape.resize(w.dim(), 1);
        w.f_reshape([w.size()[0], -1])?
            .f_norm_scalaropt_dim(2., [1], true)?
            .f_reshape(shape)?
    } else {
        let trans = calc.f_transpose(0, 1)?;
        let mut shape = vec![calc.size()[1]];
        shape.resize(calc.dim(), 1);
        trans
            .f_reshape([calc.size()[1], -1])?
            .f_norm_scalaropt_dim(2., [1], true)?
            .f_reshape(shape)?
            .f_transpose(0, 1)?
    };
    let eps = match w.kind() {
        Kind::Half => 2_f64.powi(-10),
        Kind::BFloat16 => 2_f64.powi(-7),
        Kind::Double => 2_f64.powi(-52),
        _ => 2_f64.powi(-23),
    };
    let _ = calc.f_mul_(&scale.f_div(&norm.f_add_scalar(eps)?)?.f_to_kind(w.kind())?)?;
    if p.strength != 1. {
        let _ = calc.f_sub_(w)?;
        let _ = w.f_add_(&calc.f_mul_scalar(p.strength)?)?;
    } else {
        w.f_copy_(&calc)?;
    }
    Ok(())
}
