//! Original RMS-RoPE normalization, materialization and rotary kernel.
use super::{Builder, Count, Element, Ty, Value};
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub scale: Element,
    pub frequencies: Element,
    pub block_size: u32,
    pub split_half: bool,
    pub scalars: [Count; 16],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
include!("triton_frontend_rms_rope_program.rs");
impl Builder {
    fn reduce_sum(
        &mut self,
        input: Value,
        standard_path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let stdpath = serde_json::to_string(standard_path).unwrap();
        let function = format!(
            "triton.language.standard.sum__fp32S{}S_c0_cFalse_cNone",
            self.block
        );
        if !self.helpers.iter().any(|(n, _)| n == &function) {
            let vector = input.typed(self.block);
            let body = format!(
                r#"  tt.func private @{function}(%input: {vector} loc("input"({stdpath}:287:0))) -> f32 attributes {{noinline = false}} {{
    %reduced = "tt.reduce"(%input) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %sum = tt.call @triton.language.standard._sum_combine__fp32_fp32(%a, %b) : (f32, f32) -> f32 loc({stdpath}:293:36)
      tt.reduce.return %sum : f32 loc({stdpath}:293:36)
    }}) : ({vector}) -> f32 loc({stdpath}:293:36)
    tt.return %reduced : f32 loc({stdpath}:293:11)
  ^bb1:
    %poison = ub.poison : f32 loc({stdpath}:293:4)
    tt.return %poison : f32 loc({stdpath}:293:4)
  }} loc({stdpath}:287:0)
  tt.func private @triton.language.standard._sum_combine__fp32_fp32(%a: f32 loc("a"({stdpath}:262:0)), %b: f32 loc("b"({stdpath}:262:0))) -> f32 attributes {{noinline = false}} {{
    %sum = arith.addf %a, %b : f32 loc({stdpath}:263:15)
    tt.return %sum : f32 loc({stdpath}:263:11)
  ^bb1:
    %poison = ub.poison : f32 loc({stdpath}:263:4)
    tt.return %poison : f32 loc({stdpath}:263:4)
  }} loc({stdpath}:262:0)
"#
            );
            self.helpers.push((function.clone(), body));
        }
        self.emit(
            format!(
                "tt.call @{function}({}) : ({}) -> f32",
                input.id,
                input.typed(self.block)
            ),
            Ty::Float(Element::F32),
            false,
            name,
            p,
        )
    }
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    source_ttir_scalar(s, crate::triton_frontend::scalar::ScalarType::F32)
}
pub fn source_ttir_scalar(
    s: &Specialization<'_>,
    epsilon: crate::triton_frontend::scalar::ScalarType,
) -> String {
    let path = serde_json::to_string(s.source_path).unwrap();
    let mut b = Builder {
        body: String::new(),
        next: 0,
        block: s.block_size,
        path,
        helpers: vec![],
        compute: s.frequencies,
    };
    let mut args = vec![];
    let mut pointers = vec![];
    let mut scalars = vec![];
    for (i, (name, kind)) in [
        ("x_ptr", s.input),
        ("freqs_ptr", s.frequencies),
        ("scale_ptr", s.scale),
        ("out_ptr", s.input),
    ]
    .into_iter()
    .enumerate()
    {
        let ty = Ty::Pointer(kind);
        let attr = if s.pointer_divisibility_16[i] {
            " {tt.divisibility = 16 : i32}"
        } else {
            ""
        };
        args.push(format!(
            "%{name}: {}{attr} {}",
            ty.ir(),
            b.loc(name, (10, 0))
        ));
        pointers.push(Value {
            id: format!("%{name}"),
            ty,
            shape: vec![],
            literal: None,
            float_literal: None,
            constexpr: false,
        });
    }
    for (name, count) in SCALAR_NAMES.iter().zip(s.scalars) {
        let (ty, div) = match count {
            Count::One => {
                scalars.push(Value::integer(1));
                continue;
            }
            Count::I32 { divisible_by_16 } => (Ty::I32, divisible_by_16),
            Count::I64 { divisible_by_16 } => (Ty::I64, divisible_by_16),
        };
        args.push(format!(
            "%{name}: {}{} {}",
            ty.ir(),
            if div {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            },
            b.loc(name, (10, 0))
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
    if !matches!(epsilon, crate::triton_frontend::scalar::ScalarType::One) {
        args.push(format!(
            "%epsilon: {}{} {}",
            epsilon.ty().ir(),
            if epsilon.divisible() {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            },
            b.loc("epsilon", (10, 0))
        ));
    }
    let epsilon = epsilon.value("epsilon");
    lower_rms(&mut b, s, &pointers, &scalars, epsilon);
    let helpers = b
        .helpers
        .iter()
        .map(|(_, body)| body.as_str())
        .collect::<String>();
    format!(
        "module {{\n  tt.func public @rms_rope_kernel({}) attributes {{noinline = false}} {{\n{}  }} {}\n{helpers}}} {}\n",
        args.join(", "),
        b.body,
        b.loc("", (10, 0)),
        b.loc("", (10, 0))
    )
}
