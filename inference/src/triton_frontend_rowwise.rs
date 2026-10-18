//! Typed lowering of the original rowwise INT8 quantizer and its max helper.
use super::{Count, Element, Ir};
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub input_dtype_code: u8,
    pub block_size: u32,
    pub pointer_divisibility_16: [bool; 3],
    pub count: Count,
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    let mut ir = Ir {
        body: String::new(),
        next: 0,
        block: s.block_size,
    };
    let path = serde_json::to_string(s.source_path).unwrap();
    let stdpath = serde_json::to_string(s.standard_source_path).unwrap();
    let mut args = vec![];
    for (i, (name, ty)) in [("x_ptr", s.input.ir()), ("y_ptr", "i8"), ("s_ptr", "f32")]
        .into_iter()
        .enumerate()
    {
        let attr = if s.pointer_divisibility_16[i] {
            " {tt.divisibility = 16 : i32}"
        } else {
            ""
        };
        args.push(format!(
            "%{name}: !tt.ptr<{ty}>{attr} loc(\"{name}\"({path}:772:0))"
        ));
    }
    let nty = if matches!(s.count, Count::I64 { .. }) {
        "i64"
    } else {
        "i32"
    };
    if !matches!(s.count, Count::One) {
        let attr = if matches!(
            s.count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            " {tt.divisibility = 16 : i32}"
        } else {
            ""
        };
        args.push(format!(
            "%n_elements: {nty}{attr} loc(\"n_elements\"({path}:772:0))"
        ));
    }
    let row = ir.value("tt.get_program_id x : i32".into(), "row_idx", 781, 28);
    let mut row_pointers = vec![];
    for (pointer, ty, name, line) in [
        ("%x_ptr", s.input.ir(), "x_row_ptr", 784),
        ("%y_ptr", "i8", "y_row_ptr", 785),
    ] {
        let n = if matches!(s.count, Count::One) {
            ir.value("arith.constant 1 : i32".into(), name, line, 34);
            ir.value("arith.constant 1 : i32".into(), name, line, 34)
        } else {
            "%n_elements".into()
        };
        let row = if nty == "i64" {
            ir.value(format!("arith.extsi {row} : i32 to i64"), name, line, 34)
        } else {
            row.clone()
        };
        if nty == "i32" {
            ir.overflow_temporaries("muli", &row, &n, false, name, line, 34);
        }
        let offset = ir.value(format!("arith.muli {row}, {n} : {nty}"), name, line, 34);
        row_pointers.push(ir.value(
            format!("tt.addptr {pointer}, {offset} : !tt.ptr<{ty}>, {nty}"),
            name,
            line,
            24,
        ));
    }
    let offsets = ir.value(
        format!(
            "tt.make_range {{end = {} : i32, start = 0 : i32}} : {}",
            s.block_size,
            ir.vector("i32")
        ),
        "offsets",
        788,
        27,
    );
    let compared = if nty == "i64" {
        ir.value(
            format!(
                "arith.extsi {offsets} : {} to {}",
                ir.vector("i32"),
                ir.vector("i64")
            ),
            "mask",
            789,
            21,
        )
    } else {
        offsets.clone()
    };
    let ns = if matches!(s.count, Count::One) {
        ir.value("arith.constant 1 : i32".into(), "mask", 789, 21);
        ir.value(
            format!("arith.constant dense<1> : {}", ir.vector("i32")),
            "mask",
            789,
            21,
        )
    } else {
        ir.splat("%n_elements", nty, "mask", 789, 21)
    };
    let mask = ir.value(
        format!("arith.cmpi slt, {compared}, {ns} : {}", ir.vector(nty)),
        "mask",
        789,
        21,
    );
    let xp = ir.splat(
        &row_pointers[0],
        &format!("!tt.ptr<{}>", s.input.ir()),
        "x",
        792,
        28,
    );
    let xp = ir.value(
        format!(
            "tt.addptr {xp}, {offsets} : {}, {}",
            ir.vector(&format!("!tt.ptr<{}>", s.input.ir())),
            ir.vector("i32")
        ),
        "x",
        792,
        28,
    );
    ir.value("arith.constant 0.000000e+00 : f32".into(), "x", 792, 16);
    let zero = ir.value(
        format!("arith.constant dense<0.000000e+00> : {}", ir.vector("f32")),
        "x",
        792,
        16,
    );
    let zero = ir.cast(&zero, Element::F32, s.input, true, "x", 792, 16);
    let x = ir.value(
        format!(
            "tt.load {xp}, {mask}, {zero} : {}",
            ir.vector(&format!("!tt.ptr<{}>", s.input.ir()))
        ),
        "x",
        792,
        16,
    );
    let abs = ir.value(
        format!("math.absf {x} : {}", ir.vector(s.input.ir())),
        "abs_x",
        795,
        19,
    );
    let signature = match s.input {
        Element::F32 => "fp32",
        Element::F16 => "fp16",
        Element::BF16 => "bf16",
        _ => panic!("Original registered rowwise input type"),
    };
    let maxfn = format!(
        "triton.language.standard.max__{signature}S{}S_c0_cFalse_cTrue_cFalse",
        s.block_size
    );
    let maximum = ir.value(
        format!(
            "tt.call @{maxfn}({abs}) : ({}) -> f32",
            ir.vector(s.input.ir())
        ),
        "max_val",
        798,
        21,
    );
    ir.value("arith.constant 1.270000e+02 : f32".into(), "scale", 801, 33);
    let divisor = ir.value("arith.constant 1.270000e+02 : f32".into(), "scale", 801, 33);
    let scale = ir.value(
        format!("arith.divf {maximum}, {divisor} : f32"),
        "scale",
        801,
        33,
    );
    let minimum = ir.value("arith.constant 1.000000e-30 : f32".into(), "scale", 801, 40);
    let scale = ir.value(
        format!("arith.maxnumf {scale}, {minimum} : f32"),
        "scale",
        801,
        40,
    );
    let (q, q_kind) = match s.input_dtype_code {
        1 | 2 => {
            let (kind, line, col) = if s.input_dtype_code == 1 {
                (Element::F16, 805, 44)
            } else {
                (Element::BF16, 807, 45)
            };
            let rounded = ir.cast(&scale, Element::F32, kind, false, "q_f", line, 28);
            let x = ir.cast(&x, s.input, Element::F32, true, "q_f", line, 19);
            let rounded = ir.cast(&rounded, kind, Element::F32, false, "q_f", line, 19);
            let rounded = ir.splat(&rounded, "f32", "q_f", line, 19);
            let q = ir.value(
                format!("arith.divf {x}, {rounded} : {}", ir.vector("f32")),
                "q_f",
                line,
                19,
            );
            (
                ir.cast(&q, Element::F32, kind, true, "q_f", line, col),
                kind,
            )
        }
        _ => {
            let x = ir.cast(&x, s.input, Element::F32, true, "q_f", 809, 18);
            let sp = ir.splat(&scale, "f32", "q_f", 809, 18);
            (
                ir.value(
                    format!("arith.divf {x}, {sp} : {}", ir.vector("f32")),
                    "q_f",
                    809,
                    18,
                ),
                Element::F32,
            )
        }
    };
    let q = ir.cast(&q, q_kind, Element::F32, true, "q_i", 812, 41);
    let q=ir.value(format!("tt.extern_elementwise {q} {{libname = \"\", libpath = \"\", pure = true, symbol = \"__nv_rintf\"}} : ({}) -> {}",ir.vector("f32"),ir.vector("f32")),"q_i",812,34);
    ir.value("arith.constant -1.280000e+02 : f32".into(), "q_i", 812, 63);
    ir.value("arith.constant 1.270000e+02 : f32".into(), "q_i", 812, 63);
    let lower = ir.value(
        format!("arith.constant dense<-1.280000e+02> : {}", ir.vector("f32")),
        "q_i",
        812,
        63,
    );
    let upper = ir.value(
        format!("arith.constant dense<1.270000e+02> : {}", ir.vector("f32")),
        "q_i",
        812,
        63,
    );
    let q = ir.value(
        format!(
            "tt.clampf {q}, {lower}, {upper}, propagateNan = none : {}",
            ir.vector("f32")
        ),
        "q_i",
        812,
        63,
    );
    let q = ir.value(
        format!(
            "arith.fptosi {q} : {} to {}",
            ir.vector("f32"),
            ir.vector("i32")
        ),
        "q_i",
        812,
        73,
    );
    let yp = ir.splat(&row_pointers[1], "!tt.ptr<i8>", "", 815, 25);
    let yp = ir.value(
        format!(
            "tt.addptr {yp}, {offsets} : {}, {}",
            ir.vector("!tt.ptr<i8>"),
            ir.vector("i32")
        ),
        "",
        815,
        25,
    );
    let q = ir.value(
        format!(
            "arith.trunci {q} : {} to {}",
            ir.vector("i32"),
            ir.vector("i8")
        ),
        "",
        815,
        41,
    );
    ir.line(
        format!("tt.store {yp}, {q}, {mask} : {}", ir.vector("!tt.ptr<i8>")),
        "",
        815,
        34,
    );
    let sp = ir.value(
        format!("tt.addptr %s_ptr, {row} : !tt.ptr<f32>, i32"),
        "",
        816,
        21,
    );
    ir.line(
        format!("tt.store {sp}, {scale} : !tt.ptr<f32>"),
        "",
        816,
        30,
    );
    ir.line("tt.return".into(), "", 816, 4);
    let input_type = ir.vector(s.input.ir());
    let fp32 = ir.vector("f32");
    let promotion_location = if s.input == Element::BF16 {
        "178:46"
    } else {
        "187:33"
    };
    let promoted = if s.input == Element::F32 {
        String::new()
    } else {
        format!(
            "    %promoted = arith.extf %input : {input_type} to {fp32} loc(\"input\"({stdpath}:{promotion_location}))\n"
        )
    };
    let reduced = if s.input == Element::F32 {
        "%input"
    } else {
        "%promoted"
    };
    let helpers = format!(
        r#"  tt.func private @{maxfn}(%input: {input_type} loc("input"({stdpath}:177:0))) -> f32 attributes {{noinline = false}} {{
{promoted}    %reduced = "tt.reduce"({reduced}) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %max = tt.call @triton.language.standard._elementwise_max__fp32_fp32(%a, %b) : (f32, f32) -> f32 loc({stdpath}:191:40)
      tt.reduce.return %max : f32 loc({stdpath}:191:40)
    }}) : ({fp32}) -> f32 loc({stdpath}:191:40)
    tt.return %reduced : f32 loc({stdpath}:191:15)
  ^bb1:
    %poison = ub.poison : f32 loc({stdpath}:179:4)
    tt.return %poison : f32 loc({stdpath}:179:4)
  }} loc({stdpath}:177:0)
  tt.func private @triton.language.standard._elementwise_max__fp32_fp32(%a: f32 loc("a"({stdpath}:169:0)), %b: f32 loc("b"({stdpath}:169:0))) -> f32 attributes {{noinline = false}} {{
    %max = arith.maxnumf %a, %b : f32 loc({stdpath}:170:27)
    tt.return %max : f32 loc({stdpath}:170:11)
  ^bb1:
    %poison = ub.poison : f32 loc({stdpath}:170:4)
    tt.return %poison : f32 loc({stdpath}:170:4)
  }} loc({stdpath}:169:0)
"#
    );
    format!(
        "module {{\n  tt.func public @_quantize_rowwise_kernel({}) attributes {{noinline = false}} {{\n{}  }} loc({path}:772:0)\n{helpers}}} loc({path}:772:0)\n",
        args.join(", "),
        ir.body.replace("\"@SOURCE@\"", &path)
    )
}
