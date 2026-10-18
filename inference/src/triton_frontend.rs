//! Source-positioned TTIR lowering for the frozen Kitchen FP8 elementwise bodies.
//!
//! This is a frontend component, not an installed general Triton backend. It
//! emits IR from typed scalar/pointer specializations, never saved JIT payloads.
use std::fmt::Write;
#[path = "triton_frontend_adaln.rs"]
pub mod adaln;
#[path = "triton_frontend_expr.rs"]
mod expressions;
#[path = "triton_frontend_mxfp8.rs"]
pub mod mxfp8;
#[path = "triton_frontend_rope.rs"]
pub mod rope;
#[path = "triton_frontend_rowwise.rs"]
pub mod rowwise;
#[path = "triton_frontend_scalar.rs"]
pub mod scalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    F64,
    F32,
    F16,
    BF16,
    E4M3,
    E5M2,
}
impl Element {
    fn ir(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::E4M3 => "f8E4M3FN",
            Self::E5M2 => "f8E5M2",
        }
    }
    fn fp8(self) -> bool {
        matches!(self, Self::E4M3 | Self::E5M2)
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Count {
    I32 { divisible_by_16: bool },
    I64 { divisible_by_16: bool },
    One,
}
#[derive(Clone, Copy, Debug)]
pub enum Fp8Body {
    Quantize,
    Dequantize,
}
#[derive(Clone, Copy)]
pub struct Fp8Specialization<'a> {
    pub body: Fp8Body,
    pub input: Element,
    pub output: Element,
    pub scale: Element,
    pub block_size: u32,
    pub pointer_divisibility_16: [bool; 3],
    pub count: Count,
    pub source_path: &'a str,
}
struct Ir {
    body: String,
    next: usize,
    block: u32,
}
impl Ir {
    fn value(&mut self, op: String, name: &str, line: u32, col: u32) -> String {
        let id = format!("%v{}", self.next);
        self.next += 1;
        self.line(format!("{id} = {op}"), name, line, col);
        id
    }
    fn line(&mut self, op: String, name: &str, line: u32, col: u32) {
        let pos = if line == 0 {
            "loc(unknown)".into()
        } else if name.is_empty() {
            format!("loc(\"@SOURCE@\":{line}:{col})")
        } else {
            format!("loc(\"{name}\"(\"@SOURCE@\":{line}:{col}))")
        };
        writeln!(self.body, "    {op} {pos}").unwrap();
    }
    #[allow(clippy::too_many_arguments)]
    fn overflow_temporaries(
        &mut self,
        op: &str,
        lhs: &str,
        rhs: &str,
        vector: bool,
        name: &str,
        line: u32,
        col: u32,
    ) {
        let narrow = if vector {
            self.vector("i32")
        } else {
            "i32".into()
        };
        let wide = if vector {
            self.vector("i64")
        } else {
            "i64".into()
        };
        let left = self.value(
            format!("arith.extsi {lhs} : {narrow} to {wide}"),
            name,
            line,
            col,
        );
        let right = self.value(
            format!("arith.extsi {rhs} : {narrow} to {wide}"),
            name,
            line,
            col,
        );
        let result = self.value(
            format!("arith.{op} {left}, {right} : {wide}"),
            name,
            line,
            col,
        );
        let upper = self.value("arith.constant 2147483647 : i64".into(), name, line, col);
        let lower = self.value("arith.constant -2147483648 : i64".into(), name, line, col);
        let upper = if vector {
            self.value(
                format!("arith.constant dense<2147483647> : {wide}"),
                name,
                line,
                col,
            )
        } else {
            upper
        };
        let up = self.value(
            format!("arith.cmpi sle, {result}, {upper} : {wide}"),
            name,
            line,
            col,
        );
        let lower = if vector {
            self.value(
                format!("arith.constant dense<-2147483648> : {wide}"),
                name,
                line,
                col,
            )
        } else {
            lower
        };
        let low = self.value(
            format!("arith.cmpi sge, {result}, {lower} : {wide}"),
            name,
            line,
            col,
        );
        let boolean = if vector {
            self.vector("i1")
        } else {
            "i1".into()
        };
        self.value(
            format!("arith.andi {up}, {low} : {boolean}"),
            name,
            line,
            col,
        );
    }
    fn vector(&self, ty: &str) -> String {
        format!("tensor<{}x{ty}>", self.block)
    }
    fn splat(&mut self, value: &str, ty: &str, name: &str, line: u32, col: u32) -> String {
        self.value(
            format!("tt.splat {value} : {ty} -> {}", self.vector(ty)),
            name,
            line,
            col,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn cast(
        &mut self,
        value: &str,
        from: Element,
        to: Element,
        vector: bool,
        name: &str,
        line: u32,
        col: u32,
    ) -> String {
        if from == to {
            return value.into();
        }
        let src = if vector {
            self.vector(from.ir())
        } else {
            from.ir().into()
        };
        let dst = if vector {
            self.vector(to.ir())
        } else {
            to.ir().into()
        };
        let op = if from.fp8() || to.fp8() {
            format!(
                "tt.fp_to_fp {value}{} : {src} -> {dst}",
                if to.fp8() { ", rounding = rtne" } else { "" }
            )
        } else {
            format!(
                "arith.{} {value} : {src} to {dst}",
                if to == Element::F64 || (to == Element::F32 && from != Element::F64) {
                    "extf"
                } else {
                    "truncf"
                }
            )
        };
        self.value(op, name, line, col)
    }
}
/// Lower either original body for arbitrary block size, pointer alignment,
/// input/output/scale type and original integer-count specialization.
/// Callers retain responsibility for the source wrapper's valid signature.
pub fn fp8_ttir(s: &Fp8Specialization<'_>) -> String {
    fp8_ir(s, false)
}
/// Original pre-optimization module, including dead frontend temporaries.
pub fn fp8_source_ttir(s: &Fp8Specialization<'_>) -> String {
    fp8_ir(s, true)
}
fn fp8_ir(s: &Fp8Specialization<'_>, raw: bool) -> String {
    let quant = matches!(s.body, Fp8Body::Quantize);
    let (function, first, pidline, loadline, castline, arithmeticline, storeline) = if quant {
        ("quantize_fp8_kernel_tl", 32, 40, 48, 49, 49, 52)
    } else {
        ("dequantize_fp8_kernel_tl", 103, 110, 118, 119, 119, 121)
    };
    let path = serde_json::to_string(s.source_path).unwrap();
    let mut ir = Ir {
        body: String::new(),
        next: 0,
        block: s.block_size,
    };
    let mut args = Vec::new();
    for (i, (name, kind)) in [
        ("x_ptr", s.input),
        ("output_ptr", s.output),
        ("scale_ptr", s.scale),
    ]
    .into_iter()
    .enumerate()
    {
        let attr = if s.pointer_divisibility_16[i] {
            " {tt.divisibility = 16 : i32}"
        } else {
            ""
        };
        args.push(format!(
            "%{name}: !tt.ptr<{}>{attr} loc(\"{name}\"({path}:{first}:0))",
            kind.ir()
        ));
    }
    if quant {
        args.push(format!("%lp_max: f32 loc(\"lp_max\"({path}:{first}:0))"));
    }
    let nty = match s.count {
        Count::I64 { .. } => "i64",
        _ => "i32",
    };
    let n = match s.count {
        Count::One => String::new(),
        Count::I32 { divisible_by_16 } | Count::I64 { divisible_by_16 } => {
            let attr = if divisible_by_16 {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            };
            args.push(format!(
                "%n_elements: {nty}{attr} loc(\"n_elements\"({path}:{first}:0))"
            ));
            "%n_elements".into()
        }
    };
    // Emit the canonical constants at the original TTIR boundary. The
    // source's dead overflow-check temporaries are removed by its optimizer.
    let load_zero = if raw {
        None
    } else {
        Some(ir.value(
            format!(
                "arith.constant dense<0.000000e+00> : {}",
                ir.vector(s.input.ir())
            ),
            "x",
            loadline,
            16,
        ))
    };
    let negative_zero = if quant && !raw {
        Some(ir.value("arith.constant 0.000000e+00 : f32".into(), "", 0, 0))
    } else {
        None
    };
    let mask_one = if matches!(s.count, Count::One) && !raw {
        Some(ir.value(
            format!("arith.constant dense<1> : {}", ir.vector("i32")),
            "mask",
            pidline + 3,
            21,
        ))
    } else {
        None
    };
    let (pid, block) = if raw {
        let pid = ir.value("tt.get_program_id x : i32".into(), "pid", pidline, 24);
        ir.value(
            format!("arith.constant {} : i32", s.block_size),
            "block_start",
            pidline + 1,
            24,
        );
        let block = ir.value(
            format!("arith.constant {} : i32", s.block_size),
            "block_start",
            pidline + 1,
            24,
        );
        ir.overflow_temporaries("muli", &pid, &block, false, "block_start", pidline + 1, 24);
        (pid, block)
    } else {
        let block = ir.value(format!("arith.constant {} : i32", s.block_size), "", 0, 0);
        let pid = ir.value("tt.get_program_id x : i32".into(), "pid", pidline, 24);
        (pid, block)
    };
    let start = ir.value(
        format!("arith.muli {pid}, {block} : i32"),
        "block_start",
        pidline + 1,
        24,
    );
    let range = ir.value(
        format!(
            "tt.make_range {{end = {} : i32, start = 0 : i32}} : {}",
            s.block_size,
            ir.vector("i32")
        ),
        "offsets",
        pidline + 2,
        41,
    );
    let start = ir.splat(&start, "i32", "offsets", pidline + 2, 28);
    if raw {
        ir.overflow_temporaries("addi", &start, &range, true, "offsets", pidline + 2, 28);
    }
    let offsets = ir.value(
        format!("arith.addi {start}, {range} : {}", ir.vector("i32")),
        "offsets",
        pidline + 2,
        28,
    );
    let off_mask = if nty == "i64" {
        ir.value(
            format!(
                "arith.extsi {offsets} : {} to {}",
                ir.vector("i32"),
                ir.vector("i64")
            ),
            "mask",
            pidline + 3,
            21,
        )
    } else {
        offsets.clone()
    };
    let ns = if let Some(mask) = mask_one {
        mask
    } else if matches!(s.count, Count::One) {
        ir.value("arith.constant 1 : i32".into(), "mask", pidline + 3, 21);
        ir.value(
            format!("arith.constant dense<1> : {}", ir.vector("i32")),
            "mask",
            pidline + 3,
            21,
        )
    } else {
        ir.splat(&n, nty, "mask", pidline + 3, 21)
    };
    let mask = ir.value(
        format!("arith.cmpi slt, {off_mask}, {ns} : {}", ir.vector(nty)),
        "mask",
        pidline + 3,
        21,
    );
    let scale = ir.value(
        format!("tt.load %scale_ptr : !tt.ptr<{}>", s.scale.ir()),
        "scale",
        pidline + 6,
        20,
    );
    let xp = ir.splat(
        "%x_ptr",
        &format!("!tt.ptr<{}>", s.input.ir()),
        "x",
        loadline,
        24,
    );
    let xp = ir.value(
        format!(
            "tt.addptr {xp}, {offsets} : {}, {}",
            ir.vector(&format!("!tt.ptr<{}>", s.input.ir())),
            ir.vector("i32")
        ),
        "x",
        loadline,
        24,
    );
    let load_zero = load_zero.unwrap_or_else(|| {
        ir.value(
            "arith.constant 0.000000e+00 : f32".into(),
            "x",
            loadline,
            16,
        );
        let zero = ir.value(
            format!("arith.constant dense<0.000000e+00> : {}", ir.vector("f32")),
            "x",
            loadline,
            16,
        );
        ir.cast(&zero, Element::F32, s.input, true, "x", loadline, 16)
    });
    let x = ir.value(
        format!(
            "tt.load {xp}, {mask}, {load_zero} : {}",
            ir.vector(&format!("!tt.ptr<{}>", s.input.ir()))
        ),
        "x",
        loadline,
        16,
    );
    let name = if quant { "scaled" } else { "dequantized" };
    let x = ir.cast(
        &x,
        s.input,
        Element::F32,
        true,
        name,
        castline,
        if quant { 18 } else { 23 },
    );
    let scale = ir.cast(
        &scale,
        s.scale,
        Element::F32,
        false,
        name,
        arithmeticline,
        if quant { 32 } else { 37 },
    );
    let scale = ir.splat(
        &scale,
        "f32",
        name,
        arithmeticline,
        if quant { 32 } else { 37 },
    );
    let mut result = ir.value(
        format!(
            "arith.{} {x}, {scale} : {}",
            if quant { "divf" } else { "mulf" },
            ir.vector("f32")
        ),
        name,
        arithmeticline,
        if quant { 32 } else { 37 },
    );
    if quant {
        let upper = ir.splat("%lp_max", "f32", "clamped", 51, 44);
        result = ir.value(
            format!("arith.minnumf {result}, {upper} : {}", ir.vector("f32")),
            "clamped",
            51,
            44,
        );
        let zero = negative_zero.unwrap_or_else(|| {
            ir.value(
                "arith.constant 0.000000e+00 : f32".into(),
                "clamped",
                51,
                54,
            )
        });
        let lower = ir.value(
            format!("arith.subf {zero}, %lp_max : f32"),
            "clamped",
            51,
            54,
        );
        let lower = ir.splat(&lower, "f32", "clamped", 51, 53);
        result = ir.value(
            format!("arith.maxnumf {result}, {lower} : {}", ir.vector("f32")),
            "clamped",
            51,
            53,
        );
    }
    let op = ir.splat(
        "%output_ptr",
        &format!("!tt.ptr<{}>", s.output.ir()),
        "",
        storeline,
        26,
    );
    let op = ir.value(
        format!(
            "tt.addptr {op}, {offsets} : {}, {}",
            ir.vector(&format!("!tt.ptr<{}>", s.output.ir())),
            ir.vector("i32")
        ),
        "",
        storeline,
        26,
    );
    let result = ir.cast(&result, Element::F32, s.output, true, "", storeline, 35);
    ir.line(
        format!(
            "tt.store {op}, {result}, {mask} : {}",
            ir.vector(&format!("!tt.ptr<{}>", s.output.ir()))
        ),
        "",
        storeline,
        35,
    );
    ir.line("tt.return".into(), "", storeline, 4);
    format!(
        "module {{\n  tt.func public @{function}({}) attributes {{noinline = false}} {{\n{}  }} loc({path}:{first}:0)\n}} loc({path}:{first}:0)\n",
        args.join(", "),
        ir.body.replace("\"@SOURCE@\"", &path)
    )
}

#[path = "triton_frontend_nvfp4.rs"]
pub mod nvfp4;

#[path = "triton_frontend_int8.rs"]
pub mod int8;

#[path = "triton_frontend_w4a8.rs"]
pub mod w4a8;

#[path = "triton_frontend_errors.rs"]
pub mod errors;
#[path = "triton_frontend_na.rs"]
pub mod na;
