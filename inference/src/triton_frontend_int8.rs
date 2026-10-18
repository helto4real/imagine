//! Original INT8 matmul tiling, carried pointers, integer dot and epilogues.
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
    nvfp4::build_module,
};
use std::fmt::Write;
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub per_row: bool,
    pub output: Element,
    pub weight_scale: Element,
    pub bias: Element,
    pub has_bias: bool,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub group_size_m: u32,
    pub scalars: [Count; 9],
    pub pointer_divisibility_16: [bool; 6],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
include!("triton_frontend_int8_program.rs");
struct Int8Loop {
    iv: Value,
    arguments: Vec<Value>,
    results: Vec<Value>,
    location: String,
}
impl Builder {
    fn int8_zeros(
        &mut self,
        m: Value,
        n: Value,
        standard_path: &str,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let m = u32::try_from(m.literal.unwrap()).unwrap();
        let n = u32::try_from(n.literal.unwrap()).unwrap();
        let function = format!("triton.language.standard.zeros__Tc{m}_c{n}T_ci32");
        let ty = format!("tensor<{m}x{n}xi32>");
        let path = serde_json::to_string(standard_path).unwrap();
        if !self.helpers.iter().any(|(name, _)| name == &function) {
            self.helpers.push((
                function.clone(),
                format!(
                    r#"  tt.func private @{function}() -> {ty} attributes {{noinline = false}} {{
    %zero = arith.constant 0 : i32 loc({path}:129:31)
    %zeros = arith.constant dense<0> : {ty} loc({path}:129:31)
    tt.return %zeros : {ty} loc({path}:129:11)
  ^bb1:
    %poison = ub.poison : {ty} loc({path}:129:4)
    tt.return %poison : {ty} loc({path}:129:4)
  }} loc({path}:120:0)
"#
                ),
            ));
        }
        self.emit_shape(
            format!("tt.call @{function}() : () -> {ty}"),
            Ty::I32,
            &[m, n],
            name,
            p,
        )
    }
    fn int8_dot(&mut self, a: Value, b: Value, name: &str, p: (u32, u32)) -> Value {
        let shape = [a.shape[0], b.shape[1]];
        let zero = self.constant(0, Ty::I32, false, name, p);
        let zero = self.broadcast_to(zero, &shape, name, p);
        self.emit_shape(
            format!(
                "tt.dot {}, {}, {}, inputPrecision = tf32 : {} * {} -> {}",
                a.id,
                b.id,
                zero.id,
                a.typed(self.block),
                b.typed(self.block),
                zero.typed(self.block)
            ),
            Ty::I32,
            &shape,
            name,
            p,
        )
    }
    fn begin_int8_loop(
        &mut self,
        lower: Value,
        upper: Value,
        initial: &[Value],
        origins: [(u32, u32); 3],
        p: (u32, u32),
    ) -> Int8Loop {
        let lo = self.tensor(lower, "", p);
        let hi = self.tensor(upper, "", p);
        let step = self.constant(1, Ty::I32, false, "", p);
        let iv_type = if hi.ty == Ty::I64 { Ty::I64 } else { Ty::I32 };
        let cast = |b: &mut Builder, v: Value| {
            if v.ty == iv_type {
                b.emit(
                    format!(
                        "arith.bitcast {} : {} to {}",
                        v.id,
                        iv_type.ir(),
                        iv_type.ir()
                    ),
                    iv_type,
                    false,
                    "",
                    p,
                )
            } else {
                b.cast(v, iv_type, "", p)
            }
        };
        let lo = cast(self, lo);
        let hi = cast(self, hi);
        let step = cast(self, step);
        self.emit(
            format!("ub.poison : {}", iv_type.ir()),
            iv_type,
            false,
            "",
            p,
        );
        let id = self.next;
        self.next += 1;
        let result_ids = (0..3).map(|i| format!("%loop{id}#{i}")).collect::<Vec<_>>();
        let iv = Value {
            id: format!("%iv{id}"),
            ty: iv_type,
            shape: vec![],
            literal: None,
            float_literal: None,
            constexpr: false,
        };
        let mut arguments = vec![];
        let names = ["a_ptrs", "b_ptrs", "accumulator"];
        let operand_ids = initial
            .iter()
            .map(|v| v.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.body,
            "    %loop{id}:3 = \"scf.for\"({}, {}, {}, {operand_ids}) ({{\n    ^bb0({}: {} {},",
            lo.id,
            hi.id,
            step.id,
            iv.id,
            iv_type.ir(),
            self.loc("k_idx", p)
        )
        .unwrap();
        for (i, ((v, name), origin)) in initial.iter().zip(names).zip(origins).enumerate() {
            let mut arg = v.clone();
            arg.id = format!("%carried{id}_{i}");
            let location = self.loc(name, origin);
            let location = format!(
                "loc(\"{name}\"({}))",
                location
                    .strip_prefix("loc(")
                    .unwrap()
                    .strip_suffix(')')
                    .unwrap()
            );
            writeln!(
                self.body,
                "      {}: {} {}{}",
                arg.id,
                arg.typed(self.block),
                location,
                if i == 2 { "):" } else { "," }
            )
            .unwrap();
            arguments.push(arg);
        }
        let mut location = self.loc("", p);
        for name in names {
            location = format!(
                "loc(\"{name}\"({}))",
                location
                    .strip_prefix("loc(")
                    .unwrap()
                    .strip_suffix(')')
                    .unwrap()
            );
        }
        let results = initial
            .iter()
            .zip(result_ids)
            .map(|(v, id)| Value { id, ..v.clone() })
            .collect();
        Int8Loop {
            iv,
            arguments,
            results,
            location,
        }
    }
    fn end_int8_loop(&mut self, frame: &Int8Loop, values: &[Value], p: (u32, u32)) {
        let ids = values
            .iter()
            .map(|v| v.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = values
            .iter()
            .map(|v| v.typed(self.block))
            .collect::<Vec<_>>()
            .join(", ");
        self.line(format!("scf.yield {ids} : {types}"), "", p);
        writeln!(
            self.body,
            "    }}) : ({}, {}, {}, {types}) -> ({types}) {}",
            frame.iv.ty.ir(),
            frame.iv.ty.ir(),
            frame.iv.ty.ir(),
            frame.location
        )
        .unwrap();
    }
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    build_module(
        s.source_path,
        s.block_k,
        &[
            ("a_ptr", Ty::CharPointer),
            ("b_ptr", Ty::CharPointer),
            ("c_ptr", Ty::Pointer(s.output)),
            ("a_scale_ptr", Ty::Pointer(Element::F32)),
            ("b_scale_ptr", Ty::Pointer(s.weight_scale)),
            ("bias_ptr", Ty::Pointer(s.bias)),
        ],
        &s.pointer_divisibility_16,
        &s.scalars,
        &SCALAR_NAMES,
        if s.per_row { 955 } else { 873 },
        if s.per_row {
            "_int8_matmul_dequant_per_row_kernel"
        } else {
            "_int8_matmul_dequant_kernel"
        },
        |b, p, c| {
            if s.per_row {
                lower_per_row(b, s, p, c)
            } else {
                lower_tensorwise(b, s, p, c)
            }
        },
    )
}
