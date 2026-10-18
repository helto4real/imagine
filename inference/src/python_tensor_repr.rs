//! Frozen default dense Tensor formatting for source diagnostics/cache literals.
use crate::{
    native::{Error, Kitchen},
    registry::AllowedType,
};
use tch::{Device, Kind, Tensor};
const EDGE: usize = 3;
const WIDTH: usize = 80;
#[derive(Clone)]
enum Number {
    Bool(bool),
    Int(i64),
    Float(f64),
    Complex(f64, f64),
}
fn float(n: f64, scientific: bool, integral: bool) -> String {
    if n.is_nan() {
        return "nan".into();
    }
    if n.is_infinite() {
        return if n.is_sign_negative() { "-inf" } else { "inf" }.into();
    }
    if scientific {
        let text = format!("{n:.4e}");
        let (mantissa, exponent) = text.split_once('e').unwrap();
        return format!("{mantissa}e{:+03}", exponent.parse::<i32>().unwrap());
    }
    if integral {
        format!("{n:.0}.")
    } else {
        format!("{n:.4}")
    }
}
struct Formatter {
    integral: bool,
    scientific: bool,
    width: usize,
}
impl Formatter {
    fn new(values: &[f64], floating: bool, integers: &[String]) -> Self {
        let mut out = Self {
            integral: true,
            scientific: false,
            width: 1,
        };
        if !floating {
            out.width = integers.iter().map(String::len).max().unwrap_or(1).max(1);
            return out;
        }
        let finite = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v != 0.)
            .collect::<Vec<_>>();
        if finite.is_empty() {
            return out;
        }
        let low = finite.iter().map(|n| n.abs()).fold(f64::INFINITY, f64::min);
        let high = finite.iter().map(|n| n.abs()).fold(0., f64::max);
        out.integral = finite.iter().all(|v| *v == v.ceil());
        out.scientific = high / low > 1000. || high > 1e8 || low < 1e-4;
        out.width = finite
            .into_iter()
            .map(|v| float(v, out.scientific, out.integral).len())
            .max()
            .unwrap()
            .max(1);
        out
    }
    fn render(&self, value: &Number, imaginary: bool) -> String {
        let raw = match value {
            Number::Bool(n) => if *n { "True" } else { "False" }.into(),
            Number::Int(n) => n.to_string(),
            Number::Float(n) => float(*n, self.scientific, self.integral),
            Number::Complex(re, im) => float(
                if imaginary { *im } else { *re },
                self.scientific,
                self.integral,
            ),
        };
        format!("{}{raw}", " ".repeat(self.width.saturating_sub(raw.len())))
    }
}
fn selected(size: usize, summarize: bool) -> Vec<usize> {
    if summarize && size > EDGE * 2 {
        (0..EDGE).chain(size - EDGE..size).collect()
    } else {
        (0..size).collect()
    }
}
fn sample_indices(shape: &[i64], summarize: bool) -> Vec<Vec<i64>> {
    let mut result = vec![vec![]];
    for size in shape {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                selected(*size as usize, summarize)
                    .into_iter()
                    .map(move |i| {
                        let mut p = prefix.clone();
                        p.push(i as i64);
                        p
                    })
            })
            .collect();
    }
    result
}
fn scalar(value: &Number, real: &Formatter, imag: Option<&Formatter>) -> String {
    let first = real.render(value, false);
    if let Some(imag) = imag {
        let second = format!("{}j", imag.render(value, true))
            .trim_start()
            .to_owned();
        format!(
            "{first}{}{second}",
            if second.starts_with(['+', '-']) {
                ""
            } else {
                "+"
            }
        )
    } else {
        first
    }
}
fn contents(
    shape: &[i64],
    indent: usize,
    summarize: bool,
    data: &[Number],
    cursor: &mut usize,
    real: &Formatter,
    imag: Option<&Formatter>,
) -> String {
    if shape.is_empty() {
        let value = scalar(&data[*cursor], real, imag);
        *cursor += 1;
        return value;
    }
    let indices = selected(shape[0] as usize, summarize);
    let mut slices = vec![];
    for (n, index) in indices.iter().enumerate() {
        if n > 0 && *index != indices[n - 1] + 1 {
            slices.push(if shape.len() == 1 {
                " ...".into()
            } else {
                "...".into()
            });
        }
        slices.push(contents(
            &shape[1..],
            indent + 1,
            summarize,
            data,
            cursor,
            real,
            imag,
        ));
    }
    if shape.len() == 1 {
        let element = real.width + 2 + imag.map_or(0, |f| f.width + 1);
        let per_line = ((WIDTH.saturating_sub(indent)) / element).max(1);
        let lines = slices
            .chunks(per_line)
            .map(|v| v.join(", "))
            .collect::<Vec<_>>();
        format!(
            "[{}]",
            lines.join(&format!(",\n{}", " ".repeat(indent + 1)))
        )
    } else {
        format!(
            "[{}]",
            slices.join(&format!(
                ",{}{}",
                "\n".repeat(shape.len() - 1),
                " ".repeat(indent + 1)
            ))
        )
    }
}
fn suffixes(mut text: String, values: &[String], indent: usize) -> String {
    // The original counts an absent newline as index -1.
    let mut last = text
        .rfind('\n')
        .map_or(text.len() + 2, |i| text.len() - i + 1);
    for value in values {
        if last + value.len() + 2 > WIDTH {
            text.push_str(&format!(",\n{}{value}", " ".repeat(indent)));
            last = indent + value.len();
        } else {
            text.push_str(&format!(", {value}"));
            last += value.len() + 2;
        }
    }
    text.push(')');
    text
}
fn tuple(shape: &[i64]) -> String {
    format!(
        "({}{})",
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if shape.len() == 1 { "," } else { "" }
    )
}
pub(crate) fn display(tensor: &Tensor, native: &Kitchen) -> Result<String, Error> {
    let _guard = tch::no_grad_guard();
    let default = native.default_dtype()?;
    let kind = tensor.f_kind().ok();
    let scalar_type = native.tensor_type(tensor)?.scalar_type;
    let dtype =
        match scalar_type {
            44 => "torch.float8_e8m0fnu".into(),
            45 => "torch.float4_e2m1fn_x2".into(),
            _ => AllowedType::DType(kind.ok_or_else(|| {
                Error::Native("Unknown frozen Tensor representation dtype".into())
            })?)
            .python_name()
            .to_owned(),
        };
    let complex = matches!(
        kind,
        Some(Kind::ComplexHalf | Kind::ComplexFloat | Kind::ComplexDouble)
    );
    let floating = complex
        || matches!(scalar_type, 44 | 45)
        || matches!(
            kind,
            Some(
                Kind::Half
                    | Kind::BFloat16
                    | Kind::Float
                    | Kind::Double
                    | Kind::Float8e4m3fn
                    | Kind::Float8e5m2
                    | Kind::Float8e4m3fnuz
                    | Kind::Float8e5m2fnuz
            )
        );
    let shape = tensor.size();
    let summarize = tensor.numel() > 1000;
    let mut suffix = vec![];
    if tensor.device() != Device::Cpu {
        suffix.push(format!(
            "device='{}'",
            match tensor.device() {
                Device::Cuda(i) => format!("cuda:{i}"),
                d => crate::execution_value::device_name(d),
            }
        ));
    }
    let contents = if tensor.numel() == 0 {
        if tensor.dim() != 1 {
            suffix.push(format!("size={}", tuple(&shape)));
        }
        if kind != Some(default) {
            suffix.push(format!("dtype={dtype}"));
        }
        "[]".into()
    } else {
        let default_complex = if default == Kind::Double {
            Kind::ComplexDouble
        } else {
            Kind::ComplexFloat
        };
        if ![
            Some(default),
            Some(default_complex),
            Some(Kind::Int64),
            Some(Kind::Bool),
        ]
        .contains(&kind)
        {
            suffix.push(format!("dtype={dtype}"));
        }
        let detached = tensor.f_detach()?;
        let detached = if detached.f_internal_is_zerotensor()? {
            native.clone_tensor(&detached)?
        } else {
            detached
        };
        let detached = detached.f_resolve_neg()?.f_resolve_conj()?;
        let printable = if scalar_type == 45 {
            detached.f_view_dtype(Kind::Uint8)?
        } else if matches!(scalar_type, 44 | 23 | 24 | 25 | 26) {
            detached.f_to_kind(Kind::Float)?
        } else {
            detached
        };
        let data = sample_indices(&shape, summarize)
            .into_iter()
            .map(|indices| -> Result<Number, Error> {
                Ok(if complex {
                    Number::Complex(
                        printable.f_real()?.f_double_value(&indices)?,
                        printable.f_imag()?.f_double_value(&indices)?,
                    )
                } else if kind == Some(Kind::Bool) {
                    Number::Bool(printable.f_int64_value(&indices)? != 0)
                } else if floating {
                    Number::Float(printable.f_double_value(&indices)?)
                } else {
                    Number::Int(printable.f_int64_value(&indices)?)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let floats = data
            .iter()
            .map(|n| match n {
                Number::Float(n) => *n,
                Number::Complex(re, _) => *re,
                _ => 0.,
            })
            .collect::<Vec<_>>();
        let integers = data
            .iter()
            .map(|n| match n {
                Number::Int(n) => n.to_string(),
                Number::Bool(n) => if *n { "True" } else { "False" }.into(),
                _ => String::new(),
            })
            .collect::<Vec<_>>();
        let real = Formatter::new(&floats, floating, &integers);
        let imag = if complex {
            Some(Formatter::new(
                &data
                    .iter()
                    .map(|n| {
                        if let Number::Complex(_, im) = n {
                            *im
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>(),
                true,
                &[],
            ))
        } else {
            None
        };
        contents(&shape, 7, summarize, &data, &mut 0, &real, imag.as_ref())
    };
    if let Some(name) = native.grad_fn_name(tensor)? {
        suffix.push(format!("grad_fn=<{name}>"));
    } else if tensor.requires_grad() {
        suffix.push("requires_grad=True".into());
    }
    Ok(suffixes(format!("tensor({contents}"), &suffix, 7))
}
