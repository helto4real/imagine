//! Original NVFP4 inline assembly and typed kernel source lowering.
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
};
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub global_scale: Element,
    pub block_size: u32,
    pub blocks_per_program: u32,
    pub hi_first: bool,
    pub scalars: [Count; 7],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
include!("triton_frontend_nvfp4_program.rs");
#[path = "triton_frontend_dequant_nvfp4.rs"]
pub mod dequantize;
impl Builder {
    fn bitcast(&mut self, value: Value, ty: Ty, name: &str, p: (u32, u32)) -> Value {
        if value.ty == ty {
            return value;
        }
        let target = Value {
            ty,
            ..value.clone()
        };
        self.emit_shape(
            format!(
                "tt.bitcast {} : {} -> {}",
                value.id,
                value.typed(self.block),
                target.typed(self.block)
            ),
            ty,
            &value.shape,
            name,
            p,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn inline_asm(
        &mut self,
        assembly: &str,
        constraints: &str,
        args: Vec<Value>,
        ty: Ty,
        pack: u32,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let escape = |s: &str| {
            s.bytes()
                .map(|c| {
                    if (32..=126).contains(&c) && c != b'"' && c != b'\\' {
                        (c as char).to_string()
                    } else {
                        format!("\\{c:02X}")
                    }
                })
                .collect::<String>()
        };
        let shape = args[0].shape.clone();
        let result = Value {
            id: String::new(),
            ty,
            shape: shape.clone(),
            literal: None,
            float_literal: None,
            constexpr: false,
        };
        self.emit_shape(format!("tt.elementwise_inline_asm \"{}\" {{constraints = \"{}\", packed_element = {pack} : i32, pure = true}} {} : {} -> {}",escape(assembly),escape(constraints),args.iter().map(|v|v.id.as_str()).collect::<Vec<_>>().join(", "),args.iter().map(|v|v.typed(self.block)).collect::<Vec<_>>().join(", "),result.typed(self.block)),ty,&shape,name,p)
    }
    fn reduce_nv_sum(&mut self, input: Value, path: &str, name: &str, p: (u32, u32)) -> Value {
        let rows = input.shape[0];
        let cols = input.shape[1];
        let vector = input.typed(self.block);
        let result = format!("tensor<{rows}xf32>");
        let path = serde_json::to_string(path).unwrap();
        let function = format!("triton.language.standard.sum__fp32S{rows}_{cols}S_c1_cFalse_cNone");
        if !self.helpers.iter().any(|(n, _)| n == &function) {
            let body = format!(
                r#"  tt.func private @{function}(%input: {vector} loc("input"({path}:287:0))) -> {result} attributes {{noinline = false}} {{
    %reduced = "tt.reduce"(%input) <{{axis = 1 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %sum = tt.call @triton.language.standard._sum_combine__fp32_fp32(%a, %b) : (f32, f32) -> f32 loc({path}:293:36)
      tt.reduce.return %sum : f32 loc({path}:293:36)
    }}) : ({vector}) -> {result} loc({path}:293:36)
    tt.return %reduced : {result} loc({path}:293:11)
  ^bb1:
    %poison = ub.poison : {result} loc({path}:293:4)
    tt.return %poison : {result} loc({path}:293:4)
  }} loc({path}:287:0)
  tt.func private @triton.language.standard._sum_combine__fp32_fp32(%a: f32 loc("a"({path}:262:0)), %b: f32 loc("b"({path}:262:0))) -> f32 attributes {{noinline = false}} {{
    %sum = arith.addf %a, %b : f32 loc({path}:263:15)
    tt.return %sum : f32 loc({path}:263:11)
  ^bb1:
    %poison = ub.poison : f32 loc({path}:263:4)
    tt.return %poison : f32 loc({path}:263:4)
  }} loc({path}:262:0)
"#
            );
            self.helpers.push((function.clone(), body));
        }
        self.emit_shape(
            format!("tt.call @{function}({}) : ({vector}) -> {result}", input.id),
            Ty::Float(Element::F32),
            &[rows],
            name,
            p,
        )
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn build_module(
    source: &str,
    block: u32,
    pointer_types: &[(&str, Ty)],
    align: &[bool],
    counts: &[Count],
    names: &[&str],
    definition: u32,
    function: &str,
    lower: impl FnOnce(&mut Builder, &[Value], &[Value]),
) -> String {
    let mut b = Builder {
        body: String::new(),
        next: 0,
        block,
        path: serde_json::to_string(source).unwrap(),
        helpers: vec![],
        compute: Element::F32,
    };
    let mut args = vec![];
    let mut pointers = vec![];
    let mut scalars = vec![];
    let attr = |aligned| {
        if aligned {
            " {tt.divisibility = 16 : i32}"
        } else {
            ""
        }
    };
    for (i, (name, ty)) in pointer_types.iter().enumerate() {
        args.push(format!(
            "%{name}: {}{} {}",
            ty.ir(),
            attr(align[i]),
            b.loc(name, (definition, 0))
        ));
        pointers.push(Value {
            id: format!("%{name}"),
            ty: *ty,
            shape: vec![],
            literal: None,
            float_literal: None,
            constexpr: false,
        });
    }
    for (name, count) in names.iter().zip(counts) {
        let (ty, aligned) = match count {
            Count::One => {
                scalars.push(Value::integer(1));
                continue;
            }
            Count::I32 { divisible_by_16 } => (Ty::I32, *divisible_by_16),
            Count::I64 { divisible_by_16 } => (Ty::I64, *divisible_by_16),
        };
        args.push(format!(
            "%{name}: {}{} {}",
            ty.ir(),
            attr(aligned),
            b.loc(name, (definition, 0))
        ));
        scalars.push(Value {
            id: format!("%{name}"),
            ty,
            shape: vec![],
            literal: None,
            float_literal: None,
            constexpr: false,
        });
    }
    lower(&mut b, &pointers, &scalars);
    let helpers = b
        .helpers
        .iter()
        .map(|(_, body)| body.as_str())
        .collect::<String>();
    format!(
        "module {{\n  tt.func public @{function}({}) attributes {{noinline = false}} {{\n{}  }} {}\n{helpers}}} {}\n",
        args.join(", "),
        b.body,
        b.loc("", (definition, 0)),
        b.loc("", (definition, 0))
    )
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    build_module(
        s.source_path,
        s.block_size,
        &[
            ("x_ptr", Ty::Pointer(s.input)),
            ("packed_output_ptr", Ty::BytePointer),
            ("swizzled_scales_ptr", Ty::Pointer(Element::E4M3)),
            ("per_tensor_scale_ptr", Ty::Pointer(s.global_scale)),
        ],
        &s.pointer_divisibility_16,
        &s.scalars,
        &SCALAR_NAMES,
        197,
        "quantize_nvfp4_kernel_tl",
        |b, p, c| lower_main(b, s, p, c),
    )
}

#[path = "triton_nvfp4_sources.rs"]
mod conditional_sources;
/// Frozen source compiler accepts only int, bool or None in constexpr branches.
pub fn condition_failure(
    dequantize: bool,
    kind: Option<&str>,
) -> Option<super::errors::CompilationFailure> {
    let kind = kind?;
    use conditional_sources::*;
    let (source, definition, position) = if dequantize {
        (
            DEQUANTIZE_SOURCE,
            DEQUANTIZE_DEFINITION,
            DEQUANTIZE_POSITION,
        )
    } else {
        (QUANTIZE_SOURCE, QUANTIZE_DEFINITION, QUANTIZE_POSITION)
    };
    let mut failure = super::errors::CompilationFailure::at(
        source,
        definition,
        position,
        &format!(
            "`if` conditionals can only accept values of type {{int, bool, NoneType}}, not objects of type {kind}"
        ),
    );
    failure.class = "UnsupportedLanguageConstruct";
    Some(failure)
}
