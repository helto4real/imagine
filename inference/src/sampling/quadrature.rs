//! SciPy's original QAGSE integration and NumPy's strong scalar arithmetic for
//! the LMS coefficient callback. No Python, analytic shortcut or reordered sum.
use crate::native::Error;
use std::ffi::c_void;
use tch::{Device, Kind, Tensor};

unsafe extern "C" {
    fn at_imagine_quadpack(
        integrand: extern "C" fn(f64, *mut c_void) -> f64,
        context: *mut c_void,
        a: f64,
        b: f64,
        result: *mut f64,
        error: *mut f64,
        evaluations: *mut i32,
        status: *mut i32,
    );
}
#[derive(Debug)]
pub struct Coefficient {
    pub value: f64,
    pub absolute_error: f64,
    pub evaluations: i32,
    pub status: i32,
}
impl Coefficient {
    pub fn warning(&self) -> Option<&'static str> {
        match self.status {
            0 => None,
            1 => Some(
                "The maximum number of subdivisions (50) has been achieved.\n  If increasing the limit yields no improvement it is advised to analyze \n  the integrand in order to determine the difficulties.  If the position of a \n  local difficulty can be determined (singularity, discontinuity) one will \n  probably gain from splitting up the interval and calling the integrator \n  on the subranges.  Perhaps a special-purpose integrator should be used.",
            ),
            2 => Some(
                "The occurrence of roundoff error is detected, which prevents \n  the requested tolerance from being achieved.  The error may be \n  underestimated.",
            ),
            3 => Some(
                "Extremely bad integrand behavior occurs at some points of the\n  integration interval.",
            ),
            4 => Some(
                "The algorithm does not converge.  Roundoff error is detected\n  in the extrapolation table.  It is assumed that the requested tolerance\n  cannot be achieved, and that the returned result (if full_output = 1) is \n  the best which can be obtained.",
            ),
            5 => Some("The integral is probably divergent, or slowly convergent."),
            7 => Some(
                "Abnormal termination of the routine.  The estimates for result\n  and error are less reliable.  It is assumed that the requested accuracy\n  has not been achieved.",
            ),
            _ => Some("Unknown error."),
        }
    }
}
#[derive(Clone, Copy)]
enum Arithmetic {
    Double,
    Float,
    Half,
}
impl Arithmetic {
    fn cast(self, n: f64) -> f64 {
        match self {
            Self::Double => n,
            Self::Float => n as f32 as f64,
            Self::Half => half::f16::from_f64(n).to_f64(),
        }
    }
    fn binary(self, a: f64, b: f64, op: fn(f64, f64) -> f64, op32: fn(f32, f32) -> f32) -> f64 {
        match self {
            Self::Double => op(a, b),
            Self::Float => op32(a as f32, b as f32) as f64,
            Self::Half => half::f16::from_f32(op32(a as f32, b as f32)).to_f64(),
        }
    }
    fn subtract(self, a: f64, b: f64) -> f64 {
        self.binary(a, b, |a, b| a - b, |a, b| a - b)
    }
    fn divide(self, a: f64, b: f64) -> f64 {
        self.binary(a, b, |a, b| a / b, |a, b| a / b)
    }
    fn multiply(self, a: f64, b: f64) -> f64 {
        self.binary(a, b, |a, b| a * b, |a, b| a * b)
    }
}
struct Integrand<'a> {
    sigmas: &'a [f64],
    order: usize,
    i: usize,
    j: usize,
    arithmetic: Arithmetic,
}
extern "C" fn integrand(tau: f64, context: *mut c_void) -> f64 {
    // Synchronous C callback, scoped to coefficient(); all indices checked there.
    let c = unsafe { &*(context.cast::<Integrand<'_>>()) };
    let mut product = 1.;
    for k in 0..c.order {
        if c.j != k {
            let numerator = c
                .arithmetic
                .subtract(c.arithmetic.cast(tau), c.sigmas[c.i - k]);
            let denominator = c
                .arithmetic
                .subtract(c.sigmas[c.i - c.j], c.sigmas[c.i - k]);
            product = c
                .arithmetic
                .multiply(product, c.arithmetic.divide(numerator, denominator));
        }
    }
    product
}
pub fn coefficient(order: usize, times: &Tensor, i: usize, j: usize) -> Result<Coefficient, Error> {
    if order.saturating_sub(1) > i {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!("Order {order} too high for step {i}"),
        });
    }
    let arithmetic = match times.kind() {
        Kind::Float => Arithmetic::Float,
        Kind::Double => Arithmetic::Double,
        Kind::Half => Arithmetic::Half,
        Kind::BFloat16 => {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "Got unsupported ScalarType BFloat16".into(),
            });
        }
        _ => {
            return Err(Error::BindingGap {
                backend: "lms".into(),
                operation: "non-floating NumPy sigma scalar".into(),
            });
        }
    };
    let times = times.f_to_device(Device::Cpu)?;
    let count = times.size()[0] as usize;
    if i + 1 >= count || j > i {
        return Err(Error::ReferenceException {
            class: "IndexError",
            message: "index out of bounds in LMS coefficient".into(),
        });
    }
    let sigmas = (0..count)
        .map(|i| times.double_value(&[i as i64]))
        .collect::<Vec<_>>();
    let mut context = Integrand {
        sigmas: &sigmas,
        order,
        i,
        j,
        arithmetic,
    };
    let a = sigmas[i];
    let b = sigmas[i + 1];
    // scipy.integrate.quad normalizes the bounds and restores the sign.
    let flip = b < a;
    let (mut value, mut error, mut evaluations, mut status) = (0., 0., 0, 0);
    unsafe {
        at_imagine_quadpack(
            integrand,
            (&mut context as *mut Integrand<'_>).cast(),
            a.min(b),
            a.max(b),
            &mut value,
            &mut error,
            &mut evaluations,
            &mut status,
        );
    }
    if flip {
        value = -value;
    }
    Ok(Coefficient {
        value,
        absolute_error: error,
        evaluations,
        status,
    })
}
