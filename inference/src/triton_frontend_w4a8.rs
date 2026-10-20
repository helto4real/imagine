//! Original grouped INT4/INT6 to INT8 unpacking and rounding.
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
    nvfp4::build_module,
};
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub has_cb: bool,
    pub bits: u32,
    pub block_n: u32,
    pub block_kh: u32,
    pub scalars: [Count; 9],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
}
include!("triton_frontend_w4a8_program.rs");
impl Builder {
    fn w4a8_rint(&mut self, x: Value, name: &str, p: (u32, u32)) -> Value {
        self.emit_shape(format!("tt.extern_elementwise {} {{libname = \"\", libpath = \"\", pure = true, symbol = \"__nv_rintf\"}} : ({}) -> {}",x.id,x.typed(self.block),x.typed(self.block)),x.ty,&x.shape,name,p)
    }
    fn w4a8_clamp(&mut self, x: Value, lo: Value, hi: Value, name: &str, p: (u32, u32)) -> Value {
        let x = self.tensor(x, name, p);
        let lo = self.tensor(lo, name, p);
        let hi = self.tensor(hi, name, p);
        let (x, lo) = self.broadcast_pair(x, lo, name, p);
        let (x, hi) = self.broadcast_pair(x, hi, name, p);
        self.emit_shape(
            format!(
                "tt.clampf {}, {}, {}, propagateNan = none : {}",
                x.id,
                lo.id,
                hi.id,
                x.typed(self.block)
            ),
            x.ty,
            &x.shape,
            name,
            p,
        )
    }
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    build_module(
        s.source_path,
        s.block_kh,
        &[
            ("qdata_ptr", Ty::CharPointer),
            ("srel_ptr", Ty::Pointer(Element::F32)),
            ("cb_ptr", Ty::Pointer(Element::F32)),
            ("out_ptr", Ty::CharPointer),
        ],
        &s.pointer_divisibility_16,
        &s.scalars,
        &SCALAR_NAMES,
        FUNCTION_DEFINITION,
        "_dequant_int4_grouped_to_int8_kernel",
        |b, p, c| lower_w4a8(b, s, p, c),
    )
}
