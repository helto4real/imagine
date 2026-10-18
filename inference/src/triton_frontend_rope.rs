//! Typed lowering of the frozen flexible-layout RoPE kernel.
//! The source program is translated into Rust expressions, not cached IR.
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
};
#[path = "triton_frontend_rms_rope.rs"]
pub mod rms;

#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub key: Option<Element>,
    pub frequencies: Element,
    pub block_size: u32,
    pub split_half: bool,
    pub scalars: [Count; 21],
    pub pointer_divisibility_16: [bool; 5],
    pub source_path: &'a str,
}
impl Builder {
    fn tile(&mut self, values: Vec<Value>, p: (u32, u32)) {
        let types = values
            .iter()
            .map(|v| v.typed(self.block))
            .collect::<Vec<_>>();
        let signature = values
            .iter()
            .map(|v| {
                format!(
                    "{}{}",
                    v.ty.mangle(),
                    if v.is_block() {
                        format!("S{}S", self.block)
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("_");
        let name = format!(
            "comfy_kitchen.backends.triton.rope._apply_freq_tile__{signature}_c{}",
            Ty::Float(self.compute).mangle()
        );
        if !self.helpers.iter().any(|(n, _)| n == &name) {
            let names = [
                "x_ptr",
                "x_out_ptr",
                "mask",
                "freqs_00",
                "freqs_01",
                "freqs_10",
                "freqs_11",
                "x_offset_0",
                "x_offset_1",
                "out_offset_0",
                "out_offset_1",
            ];
            let mut b = Builder {
                body: String::new(),
                next: 0,
                block: self.block,
                path: self.path.clone(),
                helpers: vec![],
                compute: self.compute,
            };
            let mut args = vec![];
            let mut helper_values = vec![];
            for ((v, n), ty) in values.iter().zip(names).zip(&types) {
                args.push(format!("%{n}: {ty} {}", b.loc(n, (145, 0))));
                let mut v = v.clone();
                v.id = format!("%{n}");
                v.literal = None;
                v.float_literal = None;
                helper_values.push(v);
            }
            lower_tile(&mut b, &helper_values);
            self.helpers.push((
                name.clone(),
                format!(
                    "  tt.func private @{name}({}) attributes {{noinline = false}} {{\n{}  }} {}\n",
                    args.join(", "),
                    b.body,
                    b.loc("", (145, 0))
                ),
            ));
        }
        self.line(
            format!(
                "tt.call @{name}({}) : ({}) -> ()",
                values
                    .iter()
                    .map(|v| v.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                types.join(", ")
            ),
            "",
            p,
        );
    }
}
include!("triton_frontend_rope_program.rs");

pub fn source_ttir(s: &Specialization<'_>) -> String {
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
    for (i, (name, kind)) in [
        ("xq_ptr", Some(s.input)),
        ("xk_ptr", s.key),
        ("freqs_ptr", Some(s.frequencies)),
        ("xq_out_ptr", Some(s.input)),
        ("xk_out_ptr", s.key),
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(kind) = kind {
            let ty = Ty::Pointer(kind);
            let attr = if s.pointer_divisibility_16[i] {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            };
            args.push(format!(
                "%{name}: {}{attr} {}",
                ty.ir(),
                b.loc(name, (24, 0))
            ));
            pointers.push(Some(Value {
                id: format!("%{name}"),
                ty,
                shape: vec![],
                literal: None,
                float_literal: None,
                constexpr: false,
            }));
        } else {
            pointers.push(None);
        }
    }
    let mut scalars = vec![];
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
            b.loc(name, (24, 0))
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
        "module {{\n  tt.func public @apply_rope_kernel({}) attributes {{noinline = false}} {{\n{}  }} {}\n{helpers}}} {}\n",
        args.join(", "),
        b.body,
        b.loc("", (24, 0)),
        b.loc("", (24, 0))
    )
}
