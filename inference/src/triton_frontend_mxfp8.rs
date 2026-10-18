//! Source-positioned MXFP8 quantization and blocked scale indexing.
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
};
use std::fmt::Write;
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub block_size: u32,
    pub blocks_per_program: u32,
    pub scalars: [Count; 6],
    pub pointer_divisibility_16: [bool; 3],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
include!("triton_frontend_mxfp8_program.rs");
impl Builder {
    pub(super) fn begin_for(
        &mut self,
        upper: i64,
        name: &str,
        p: (u32, u32),
    ) -> ((u32, u32), Value) {
        let lo = self.constant(0, Ty::I32, false, "", p);
        let hi = self.constant(upper, Ty::I32, false, "", p);
        let step = self.constant(1, Ty::I32, false, "", p);
        let cast = |b: &mut Builder, v: Value| {
            b.emit(
                format!("arith.bitcast {} : i32 to i32", v.id),
                Ty::I32,
                false,
                "",
                p,
            )
        };
        let lo = cast(self, lo);
        let hi = cast(self, hi);
        let step = cast(self, step);
        self.emit("ub.poison : i32".into(), Ty::I32, false, "", p);
        let id = format!("%iv{}", self.next);
        self.next += 1;
        writeln!(
            self.body,
            "    \"scf.for\"({}, {}, {}) ({{\n    ^bb0({id}: i32 {}):",
            lo.id,
            hi.id,
            step.id,
            self.loc(name, p)
        )
        .unwrap();
        (
            p,
            Value {
                id,
                ty: Ty::I32,
                shape: vec![],
                literal: None,
                float_literal: None,
                constexpr: false,
            },
        )
    }
    pub(super) fn end_for(&mut self, p: (u32, u32)) {
        self.line("scf.yield".into(), "", p);
        writeln!(
            self.body,
            "    }}) : (i32, i32, i32) -> () {}",
            self.loc("", p)
        )
        .unwrap();
    }
    pub(super) fn begin_if(&mut self, condition: Value, p: (u32, u32)) -> (u32, u32) {
        writeln!(self.body, "    scf.if {} {{", condition.id).unwrap();
        p
    }
    pub(super) fn end_if(&mut self, p: (u32, u32)) {
        self.line("scf.yield".into(), "", p);
        self.body.push_str("    } else {\n");
        self.line("scf.yield".into(), "", p);
        writeln!(self.body, "    }} {}", self.loc("", p)).unwrap();
    }
    pub(super) fn reduce_max(
        &mut self,
        input: Value,
        standard_path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let path = serde_json::to_string(standard_path).unwrap();
        let function = format!(
            "triton.language.standard.max__fp32S{}S_c0_cFalse_cTrue_cFalse",
            self.block
        );
        if !self.helpers.iter().any(|(n, _)| n == &function) {
            let vector = input.typed(self.block);
            let body = format!(
                r#"  tt.func private @{function}(%input: {vector} loc("input"({path}:177:0))) -> f32 attributes {{noinline = false}} {{
    %reduced = "tt.reduce"(%input) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %max = tt.call @triton.language.standard._elementwise_max__fp32_fp32(%a, %b) : (f32, f32) -> f32 loc({path}:191:40)
      tt.reduce.return %max : f32 loc({path}:191:40)
    }}) : ({vector}) -> f32 loc({path}:191:40)
    tt.return %reduced : f32 loc({path}:191:15)
  ^bb1:
    %poison = ub.poison : f32 loc({path}:179:4)
    tt.return %poison : f32 loc({path}:179:4)
  }} loc({path}:177:0)
  tt.func private @triton.language.standard._elementwise_max__fp32_fp32(%a: f32 loc("a"({path}:169:0)), %b: f32 loc("b"({path}:169:0))) -> f32 attributes {{noinline = false}} {{
    %max = arith.maxnumf %a, %b : f32 loc({path}:170:27)
    tt.return %max : f32 loc({path}:170:11)
  ^bb1:
    %poison = ub.poison : f32 loc({path}:170:4)
    tt.return %poison : f32 loc({path}:170:4)
  }} loc({path}:169:0)
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
    fn integer_helper(
        &mut self,
        which: &str,
        values: Vec<Value>,
        source_path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let (names, line, ret_line, qualification) = if which == "cdiv" {
            (&["x", "div"][..], 34, 43, "triton.language.standard.cdiv")
        } else {
            (
                &["in_row", "in_col", "n_col_blocks", "padded_scale_cols"][..],
                161,
                193,
                "comfy_kitchen.backends.triton.quantization._compute_swizzled_scale_offset",
            )
        };
        let signature = values
            .iter()
            .map(|v| {
                if v.constexpr {
                    format!("c{}", v.literal.unwrap())
                } else {
                    if v.shape.is_empty() {
                        v.ty.mangle()
                    } else {
                        format!(
                            "{}S{}S",
                            v.ty.mangle(),
                            v.shape
                                .iter()
                                .map(u32::to_string)
                                .collect::<Vec<_>>()
                                .join("_")
                        )
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("_");
        let function = format!("{qualification}__{signature}");
        let result_ty = if values.iter().any(|v| v.ty == Ty::I64) {
            Ty::I64
        } else {
            Ty::I32
        };
        let result_shape = values
            .iter()
            .max_by_key(|v| v.shape.len())
            .unwrap()
            .shape
            .clone();
        let result_template = Value {
            id: String::new(),
            ty: result_ty,
            shape: result_shape.clone(),
            literal: None,
            float_literal: None,
            constexpr: false,
        };
        let result_type = result_template.typed(self.block);
        let runtime = values.iter().filter(|v| !v.constexpr).collect::<Vec<_>>();
        let constexpr_result = if which == "cdiv" && values.iter().all(|v| v.constexpr) {
            Some(Value::integer(
                (values[0].literal.unwrap() + values[1].literal.unwrap() - 1)
                    / values[1].literal.unwrap(),
            ))
        } else {
            None
        };
        if !self.helpers.iter().any(|(n, _)| n == &function) {
            let mut b = Builder {
                body: String::new(),
                next: 0,
                block: self.block,
                path: serde_json::to_string(source_path).unwrap(),
                helpers: vec![],
                compute: Element::F32,
            };
            let mut arguments = vec![];
            let mut bound = vec![];
            for (v, arg) in values.iter().zip(names) {
                if v.constexpr {
                    bound.push(v.clone());
                    continue;
                }
                arguments.push(format!(
                    "%{arg}: {} {}",
                    v.typed(self.block),
                    b.loc(arg, (line, 0))
                ));
                let mut v = v.clone();
                v.id = format!("%{arg}");
                v.literal = None;
                v.float_literal = None;
                bound.push(v);
            }
            let result = if which == "cdiv" {
                lower_cdiv(&mut b, &bound)
            } else {
                lower_swizzle(&mut b, &bound)
            };
            assert!(result.ty == result_ty);
            b.body.push_str("  ^bb1:\n");
            if result.constexpr {
                b.line("tt.return".into(), "", (ret_line, 4));
            } else {
                let poison = b.emit_shape(
                    format!("ub.poison : {result_type}"),
                    result_ty,
                    &result_shape,
                    "",
                    (ret_line, 4),
                );
                b.line(
                    format!("tt.return {} : {result_type}", poison.id),
                    "",
                    (ret_line, 4),
                );
            }
            self.helpers.push((function.clone(),format!("  tt.func private @{function}({}){} attributes {{noinline = false}} {{\n{}  }} {}\n",arguments.join(", "),if result.constexpr{String::new()}else{format!(" -> {result_type}")},b.body,b.loc("",(line,0)))));
        }
        let call = format!(
            "tt.call @{function}({}) : ({}) -> {}",
            runtime
                .iter()
                .map(|v| v.id.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            runtime
                .iter()
                .map(|v| v.typed(self.block))
                .collect::<Vec<_>>()
                .join(", "),
            if constexpr_result.is_some() {
                "()".into()
            } else {
                result_type.clone()
            }
        );
        if let Some(value) = constexpr_result {
            self.line(call, name, p);
            value
        } else {
            self.emit_shape(call, result_ty, &result_shape, name, p)
        }
    }

    pub(super) fn cdiv(
        &mut self,
        x: Value,
        div: Value,
        path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        self.integer_helper("cdiv", vec![x, div], path, name, p)
    }
    pub(super) fn swizzle(
        &mut self,
        args: Vec<Value>,
        path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        self.integer_helper("swizzle", args, path, name, p)
    }
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    let mut b = Builder {
        body: String::new(),
        next: 0,
        block: s.block_size,
        path: serde_json::to_string(s.source_path).unwrap(),
        helpers: vec![],
        compute: Element::F32,
    };
    let mut args = vec![];
    let mut pointers = vec![];
    let mut scalars = vec![];
    for (i, (name, ty)) in [
        ("x_ptr", Ty::Pointer(s.input)),
        ("output_ptr", Ty::Pointer(Element::E4M3)),
        ("swizzled_scales_ptr", Ty::BytePointer),
    ]
    .into_iter()
    .enumerate()
    {
        args.push(format!(
            "%{name}: {}{} {}",
            ty.ir(),
            if s.pointer_divisibility_16[i] {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            },
            b.loc(name, (586, 0))
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
        let (ty, aligned) = match count {
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
            if aligned {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            },
            b.loc(name, (586, 0))
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
    lower_main(&mut b, s, &pointers, &scalars);
    let helpers = b
        .helpers
        .iter()
        .map(|(_, body)| body.as_str())
        .collect::<String>();
    format!(
        "module {{\n  tt.func public @quantize_mxfp8_kernel_tl({}) attributes {{noinline = false}} {{\n{}  }} {}\n{helpers}}} {}\n",
        args.join(", "),
        b.body,
        b.loc("", (586, 0)),
        b.loc("", (586, 0))
    )
}
