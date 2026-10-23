//! Ordered binary64 instructions at the locked Linux/x86-64 CPython boundary.
//! LLVM may commute a Rust fadd and change which NaN payload survives. Keep the
//! reference instruction's source/destination order, including MXCSR behavior.
use crate::native::Error;
pub(crate) fn add(mut left: f64, right: f64) -> Result<f64, Error> {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::asm!("addsd {left}, {right}",left=inout(xmm_reg)left,right=in(xmm_reg)right,options(nostack,nomem));
        }
        Ok(left)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (left, right);
        Err(Error::BindingGap {
            backend: "cpython_float".into(),
            operation: "ordered binary64 add requires a verified platform binding".into(),
        })
    }
}
pub(crate) fn sub(mut left: f64, right: f64) -> Result<f64, Error> {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::asm!("subsd {left}, {right}",left=inout(xmm_reg)left,right=in(xmm_reg)right,options(nostack,nomem));
        }
        Ok(left)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (left, right);
        Err(Error::BindingGap {
            backend: "cpython_float".into(),
            operation: "ordered binary64 sub requires a verified platform binding".into(),
        })
    }
}
