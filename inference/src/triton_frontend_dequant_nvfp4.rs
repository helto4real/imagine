//! Frozen inverse NVFP4 conversion, nibble order and scale addressing.
use super::{Builder, Count, Element, Ty, Value, build_module};
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub output: Element,
    pub block_scale: Element,
    pub global_scale: Element,
    pub block_size: u32,
    pub tile_size: u32,
    pub hi_first: bool,
    pub scalars: [Count; 4],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
}
include!("triton_frontend_dequant_nvfp4_program.rs");
pub fn source_ttir(s: &Specialization<'_>) -> String {
    build_module(
        s.source_path,
        s.tile_size,
        &[
            ("packed_ptr", Ty::BytePointer),
            ("scale_ptr", Ty::Pointer(s.block_scale)),
            ("global_scale_ptr", Ty::Pointer(s.global_scale)),
            ("output_ptr", Ty::Pointer(s.output)),
        ],
        &s.pointer_divisibility_16,
        &s.scalars,
        &SCALAR_NAMES,
        420,
        "dequantize_nvfp4_kernel_tl",
        |b, p, c| lower_main(b, s, p, c),
    )
}
