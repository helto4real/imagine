//! Align Your Steps / GITS source tables and NumPy's log-linear interpolation.
use crate::native::Error;
use serde_json::Value;
use tch::Tensor;
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn linspace(n: i64) -> Result<Vec<f64>, Error> {
    if n < 0 {
        return Err(error(
            "ValueError",
            format!("Number of samples, {n}, must be non-negative."),
        ));
    }
    let mut values = (0..n).map(|i| i as f64).collect::<Vec<_>>();
    if n > 1 {
        let step = 1. / (n - 1) as f64;
        for v in &mut values {
            *v *= step;
        }
        values[(n - 1) as usize] = 1.;
    }
    Ok(values)
}
fn loglinear_values(values: &[f64], n: i64) -> Result<Vec<f64>, Error> {
    let x = linspace(values.len() as i64)?;
    let new = linspace(n)?;
    if values.is_empty() {
        return Err(error("ValueError", "array of sample points is empty"));
    }
    let y = values
        .iter()
        .rev()
        .map(|v| super::numpy_scalar::log_f64_numpy(*v))
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    for point in new {
        let value = if values.len() == 1 || point <= x[0] {
            y[0]
        } else if point >= x[x.len() - 1] {
            y[y.len() - 1]
        } else {
            let index = x.partition_point(|old| *old <= point) - 1;
            if point == x[index] {
                y[index]
            } else {
                let slope = (y[index + 1] - y[index]) / (x[index + 1] - x[index]);
                let mut result = slope * (point - x[index]) + y[index];
                if result.is_nan() {
                    result = slope * (point - x[index + 1]) + y[index + 1];
                    if result.is_nan() && y[index] == y[index + 1] {
                        result = y[index];
                    }
                }
                result
            }
        };
        out.push(value.exp());
    }
    out.reverse();
    Ok(out)
}
pub fn loglinear(values: &[f64], n: i64) -> Result<Tensor, Error> {
    let values = loglinear_values(values, n)?;
    let tensor = Tensor::f_from_slice(&values)?;
    // NumPy's final dense copy has zero stride for an empty 1D array.
    Ok(if values.is_empty() {
        tensor.f_as_strided([0], [0], Some(0))?
    } else {
        tensor
    })
}
fn tail(mut values: Vec<f64>, count: i64) -> Vec<f64> {
    let len = values.len() as i64;
    // Source [-(total_steps + 1):], including the unusual zero/negative cases.
    let start = if count > 0 {
        (len - count).max(0)
    } else {
        (-count).min(len)
    };
    values.drain(..start as usize);
    values
}
fn finish(mut values: Vec<f64>, count: i64) -> Result<Tensor, Error> {
    values = tail(values, count);
    let Some(last) = values.last_mut() else {
        return Err(error(
            "IndexError",
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    };
    *last = 0.;
    Ok(Tensor::f_from_slice(
        &values.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
    )?)
}
fn total(steps: i64, denoise: f64) -> Option<i64> {
    if denoise < 1. {
        if denoise <= 0. {
            None
        } else {
            Some((steps as f64 * denoise).round_ties_even() as i64)
        }
    } else {
        Some(steps)
    }
}
fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
pub fn align_your_steps(model: &str, steps: i64, denoise: f64) -> Result<Tensor, Error> {
    let Some(count) = total(steps, denoise) else {
        return Ok(Tensor::f_from_slice::<f32>(&[])?);
    };
    let data: Value =
        serde_json::from_str(include_str!("../../resources/shared-schedule-tables.json"))
            .map_err(|e| Error::Native(e.to_string()))?;
    let table = data["ays"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row[0] == model)
        .ok_or_else(|| error("KeyError", format!("'{model}'")))?;
    let mut values = floats(&table[1]);
    if steps + 1 != values.len() as i64 {
        values = loglinear_values(&values, steps + 1)?;
    }
    finish(values, count + 1)
}
pub fn gits(coefficient: f64, steps: i64, denoise: f64) -> Result<Tensor, Error> {
    let Some(count) = total(steps, denoise) else {
        return Ok(Tensor::f_from_slice::<f32>(&[])?);
    };
    let rounded = format!("{coefficient:.2}")
        .parse::<f64>()
        .map_err(|e| Error::Native(e.to_string()))?;
    let data: Value =
        serde_json::from_str(include_str!("../../resources/shared-schedule-tables.json"))
            .map_err(|e| Error::Native(e.to_string()))?;
    let table = data["gits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row[0].as_f64() == Some(rounded))
        .ok_or_else(|| error("KeyError", crate::registry::python_float(rounded)))?;
    let rows = table[1].as_array().unwrap();
    let index = if steps <= 20 {
        let index = steps - 2;
        if index < 0 {
            rows.len() as i64 + index
        } else {
            index
        }
    } else {
        rows.len() as i64 - 1
    };
    let row = rows
        .get(index as usize)
        .ok_or_else(|| error("IndexError", "list index out of range"))?;
    let mut values = floats(row);
    if steps > 20 {
        values = loglinear_values(&values, steps + 1)?;
    }
    finish(values, count + 1)
}
