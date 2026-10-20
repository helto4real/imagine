//! Original DEIS coefficient integration, including its autograd calculation.
use super::{locals::SourceLocals, solvers::Execution};
use crate::native::{self, Error, Kitchen};
use tch::{Device, Tensor};
#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    Integer(i64),
    Float(f64),
}
fn numpy_log_of_squared_plus_one(
    native: &Kitchen,
    value: Scalar,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    // The source's default sigma_max is integer 80. np.log(int64) returns
    // float64, while np.log(torch.tensor(80.0)) retains its floating dtype.
    let tensor = match value {
        Scalar::Integer(n) => Tensor::from(n),
        Scalar::Float(n) => Tensor::from(n).f_to_kind(native.default_dtype()?)?,
    };
    let tensor = tensor.f_pow_tensor_scalar(2)?.f_add_scalar(1)?;
    let tensor = if matches!(value, Scalar::Integer(_)) {
        tensor.f_to_kind(tch::Kind::Double)?
    } else {
        tensor
    };
    let result = super::numpy_scalar::log_scalar(&tensor)?;
    warning(
        "__array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)",
    );
    Ok(result)
}

pub fn edm_to_time(
    native: &Kitchen,
    steps: &Tensor,
    epsilon: f64,
    min: Scalar,
    max: Scalar,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    edm_to_time_with_diagnostics(native, steps, epsilon, min, max, &mut |_| {})
}
fn edm_to_time_with_diagnostics(
    native: &Kitchen,
    steps: &Tensor,
    epsilon: f64,
    min: Scalar,
    max: Scalar,
    warning: &mut dyn FnMut(&str),
) -> Result<(Tensor, Tensor, Tensor), Error> {
    let min = numpy_log_of_squared_plus_one(native, min, warning)?.f_div_scalar(epsilon)?;
    let max_log = numpy_log_of_squared_plus_one(native, max, warning)?;
    let beta_d = min
        .f_sub(&max_log)?
        .f_mul_scalar(2)?
        .f_div_scalar(epsilon - 1.)?;
    // Original calls np.log(max) again after finishing beta_d. Preserve that
    // operation and its diagnostic rather than reusing the first result.
    let beta_min =
        numpy_log_of_squared_plus_one(native, max, warning)?.f_sub(&beta_d.f_mul_scalar(0.5)?)?;
    let time_beta_d = native::clone_tensor(&beta_d)?
        .f_detach()?
        .f_to_device(Device::Cpu)?;
    let time_beta_min = native::clone_tensor(&beta_min)?
        .f_detach()?
        .f_to_device(Device::Cpu)?;
    let steps = native::clone_tensor(steps)?
        .f_detach()?
        .f_to_device(Device::Cpu)?;
    let time = time_beta_min
        .f_pow_tensor_scalar(2)?
        .f_add(
            &time_beta_d
                .f_mul_scalar(2)?
                .f_mul(&steps.f_pow_tensor_scalar(2)?.f_add_scalar(1)?.f_log()?)?,
        )?
        .f_sqrt()?
        .f_sub(&time_beta_min)?
        .f_div(&time_beta_d)?;
    Ok((time, beta_min.shallow_clone(), beta_d.f_add(&beta_min)?))
}
pub fn integrand(beta0: &Tensor, beta1: &Tensor, taus: &Tensor) -> Result<Tensor, Error> {
    let (alpha, gradient) = {
        let _inference = native::inference_mode(false)?;
        let copy = |t: &Tensor| native::clone_tensor(t);
        let taus = copy(taus)?;
        let beta0 = copy(beta0)?;
        let beta1 = copy(beta1)?;
        tch::with_grad(|| -> Result<(Tensor, Tensor), Error> {
            let taus = taus.f_set_requires_grad(true)?;
            let alpha = taus
                .f_pow_tensor_scalar(2)?
                .f_mul_scalar(-0.5)?
                .f_mul(&beta1.f_sub(&beta0)?)?
                .f_sub(&taus.f_mul(&beta0)?)?
                .f_exp()?;
            alpha.f_log()?.f_sum(None)?.f_backward()?;
            Ok((alpha, taus.f_grad()?))
        })?
    };
    Ok(gradient
        .f_mul_scalar(-0.5)?
        .f_div(&alpha.f_mul(&alpha.f_rsub_scalar(1)?)?.f_sqrt()?)?)
}
fn polynomial(previous: &Tensor, j: i64, taus: &Tensor) -> Result<Tensor, Error> {
    let mut poly: Option<Tensor> = None;
    for k in 0..previous.size()[0] {
        if k != j {
            let term = taus
                .f_sub(&previous.f_get(k)?)?
                .f_div(&previous.f_get(j)?.f_sub(&previous.f_get(k)?)?)?;
            if let Some(poly) = &mut poly {
                let _ = poly.f_mul_(&term)?;
            } else {
                poly = Some(term.f_mul_scalar(1)?);
            }
        }
    }
    Ok(poly.unwrap_or(Tensor::from(1)))
}
fn integral2(
    a: &Tensor,
    b: &Tensor,
    start: &Tensor,
    end: &Tensor,
    c: &Tensor,
) -> Result<Tensor, Error> {
    let coefficient = end
        .f_pow_tensor_scalar(3)?
        .f_sub(&start.f_pow_tensor_scalar(3)?)?
        .f_div_scalar(3)?
        .f_sub(
            &end.f_pow_tensor_scalar(2)?
                .f_sub(&start.f_pow_tensor_scalar(2)?)?
                .f_mul(&a.f_add(b)?)?
                .f_div_scalar(2)?,
        )?
        .f_add(&end.f_sub(start)?.f_mul(a)?.f_mul(b)?)?;
    Ok(coefficient.f_div(&c.f_sub(a)?.f_mul(&c.f_sub(b)?)?)?)
}
fn integral3(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    start: &Tensor,
    end: &Tensor,
    d: &Tensor,
) -> Result<Tensor, Error> {
    let coefficient = end
        .f_pow_tensor_scalar(4)?
        .f_sub(&start.f_pow_tensor_scalar(4)?)?
        .f_div_scalar(4)?
        .f_sub(
            &end.f_pow_tensor_scalar(3)?
                .f_sub(&start.f_pow_tensor_scalar(3)?)?
                .f_mul(&a.f_add(b)?.f_add(c)?)?
                .f_div_scalar(3)?,
        )?
        .f_add(
            &end.f_pow_tensor_scalar(2)?
                .f_sub(&start.f_pow_tensor_scalar(2)?)?
                .f_mul(&a.f_mul(b)?.f_add(&a.f_mul(c)?)?.f_add(&b.f_mul(c)?)?)?
                .f_div_scalar(2)?,
        )?
        .f_sub(&end.f_sub(start)?.f_mul(a)?.f_mul(b)?.f_mul(c)?)?;
    Ok(coefficient.f_div(&d.f_sub(a)?.f_mul(&d.f_sub(b)?)?.f_mul(&d.f_sub(c)?)?)?)
}
pub fn coefficients(
    native: &Kitchen,
    steps: &Tensor,
    max_order: i64,
    n: i64,
    mode: &str,
) -> Result<Vec<Vec<Tensor>>, Error> {
    coefficients_with_diagnostics(native, steps, max_order, n, mode, &mut |_| {})
}
fn coefficients_with_diagnostics(
    native: &Kitchen,
    steps: &Tensor,
    max_order: i64,
    n: i64,
    mode: &str,
    warning: &mut dyn FnMut(&str),
) -> Result<Vec<Vec<Tensor>>, Error> {
    let mut output = Vec::new();
    match mode {
        "tab" => {
            let (time, beta0, beta1) = edm_to_time_with_diagnostics(
                native,
                steps,
                1e-3,
                Scalar::Float(0.002),
                Scalar::Integer(80),
                warning,
            )?;
            for i in 0..time.size()[0] - 1 {
                let current = time.f_get(i)?;
                let next = time.f_get(i + 1)?;
                let order = (i + 1).min(max_order);
                if order == 1 {
                    output.push(Vec::new());
                    continue;
                }
                let taus = Tensor::f_linspace_tensor_tensor(
                    &current,
                    &next,
                    n,
                    (native.default_dtype()?, Device::Cpu),
                )?;
                let dtau = next.f_sub(&current)?.f_div_scalar(n)?;
                let indices = Tensor::f_from_slice(&(0..order).map(|k| i - k).collect::<Vec<_>>())?;
                let previous = time.f_index_select(0, &indices)?;
                let integrand = integrand(&beta0, &beta1, &taus)?;
                let mut coefficients = Vec::new();
                for j in 0..order {
                    coefficients.push(
                        integrand
                            .f_mul(&polynomial(&previous, j, &taus)?)?
                            .f_sum(None)?
                            .f_mul(&dtau)?,
                    );
                }
                output.push(coefficients);
            }
        }
        "rhoab" => {
            for i in 0..steps.size()[0] - 1 {
                let current = steps.f_get(i)?;
                let next = steps.f_get(i + 1)?;
                let order = i.min(max_order);
                let p = (0..=order)
                    .map(|k| steps.f_get(i - k).map_err(Error::from))
                    .collect::<Result<Vec<_>, _>>()?;
                output.push(match order{
                    0=>Vec::new(),
                    1=>vec![next.f_sub(&p[1])?.f_pow_tensor_scalar(2)?.f_sub(&current.f_sub(&p[1])?.f_pow_tensor_scalar(2)?)?.f_div(&current.f_sub(&p[1])?.f_mul_scalar(2)?)?,next.f_sub(&current)?.f_pow_tensor_scalar(2)?.f_div(&p[1].f_sub(&current)?.f_mul_scalar(2)?)?],
                    2=>vec![integral2(&p[1],&p[2],&current,&next,&current)?,integral2(&current,&p[2],&current,&next,&p[1])?,integral2(&current,&p[1],&current,&next,&p[2])?],
                    3=>vec![integral3(&p[1],&p[2],&p[3],&current,&next,&current)?,integral3(&current,&p[2],&p[3],&current,&next,&p[1])?,integral3(&current,&p[1],&p[3],&current,&next,&p[2])?,integral3(&current,&p[1],&p[2],&current,&next,&p[3])?],
                    _=>return Err(Error::ReferenceException{class:"UnboundLocalError",message:"cannot access local variable 'coeff_temp' where it is not associated with a value".into()}),
                });
            }
        }
        _ => {
            return Err(Error::ReferenceException {
                class: "UnboundLocalError",
                message: "cannot access local variable 'C' where it is not associated with a value"
                    .into(),
            });
        }
    }
    Ok(output)
}
pub fn sample(
    exec: &mut dyn Execution,
    native: &Kitchen,
    x: Tensor,
    sigmas: &Tensor,
    max_order: i64,
    mode: &str,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    let _ = frame.keep("x", x.shallow_clone());
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let mut next_x = frame.keep("x_next", x.shallow_clone());
    let _ = frame.keep("t_steps", sigmas.shallow_clone());
    let coefficients =
        coefficients_with_diagnostics(native, sigmas, max_order, 10000, mode, &mut |message| {
            exec.warning(message)
        })?;
    for (index, coefficients) in coefficients.iter().enumerate() {
        frame.keep_list(&format!("coeff_list[{index}]"), coefficients);
    }
    let mut history: Vec<Tensor> = Vec::new();
    for i in 0..sigmas.size()[0] - 1 {
        let current = frame.keep("t_cur", sigmas.f_get(i)?);
        let next = frame.keep("t_next", sigmas.f_get(i + 1)?);
        let current_x = frame.keep("x_cur", next_x.shallow_clone());
        let p = frame.denoise(exec, &current_x, &current.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &x,
                i,
                sigma: &current,
                sigma_hat: &current,
                denoised: &p.denoised,
            },
        )?;
        let d = frame.keep("d_cur", current_x.f_sub(&p.denoised)?.f_div(&current)?);
        let order = if next.double_value(&[]) <= 0. {
            1
        } else {
            max_order.min(i + 1)
        };
        if order == 1 {
            next_x = current_x.f_add(&next.f_sub(&current)?.f_mul(&d)?)?;
        } else if (2..=4).contains(&order) {
            let coefficients = &coefficients[i as usize];
            if coefficients.len() as i64 != order {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: if coefficients.len() as i64 > order {
                        format!("too many values to unpack (expected {order})")
                    } else {
                        format!(
                            "not enough values to unpack (expected {order}, got {})",
                            coefficients.len()
                        )
                    },
                });
            }
            for (index, coefficient) in coefficients.iter().enumerate() {
                let _ = frame.keep(
                    if index == 0 {
                        "coeff_cur".into()
                    } else {
                        format!("coeff_prev{index}")
                    },
                    coefficient.shallow_clone(),
                );
            }
            next_x = current_x.f_add(&coefficients[0].f_mul(&d)?)?;
            for (c, h) in coefficients[1..].iter().zip(history.iter().rev()) {
                next_x = next_x.f_add(&c.f_mul(h)?)?;
            }
        }
        let _ = frame.keep("x_next", next_x.shallow_clone());
        if history.len() as i64 == max_order - 1 {
            if history.is_empty() {
                return Err(Error::ReferenceException {
                    class: "IndexError",
                    message: "list assignment index out of range".into(),
                });
            }
            let _ = history.remove(0);
            history.push(d.f_detach()?);
        } else {
            history.push(d.f_detach()?);
        }
        frame.keep_list("buffer_model", &history);
    }
    frame.finish(exec, next_x)
}
