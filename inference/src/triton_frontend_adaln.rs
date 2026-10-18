//! Original two-pass AdaLN and RMS-AdaLN, including typed loop arguments.
use super::{Count, Element, Ir};
use std::fmt::Write;
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub scale: Element,
    pub shift: Element,
    pub block_size: u32,
    pub subtract_mean: bool,
    /// d, scale_group, shift_group, stride_xr, stride_sr, stride_shr, stride_yr.
    pub scalars: [Count; 7],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
#[derive(Clone)]
struct Index {
    v: String,
    wide: bool,
}
impl Index {
    fn ty(&self) -> &'static str {
        if self.wide { "i64" } else { "i32" }
    }
}
const NAMES: [&str; 7] = [
    "d",
    "scale_group",
    "shift_group",
    "stride_xr",
    "stride_sr",
    "stride_shr",
    "stride_yr",
];
fn wide(c: Count) -> bool {
    matches!(c, Count::I64 { .. })
}
#[allow(clippy::too_many_arguments)]
fn bin_arg(
    ir: &mut Ir,
    left: &Index,
    index: usize,
    s: &Specialization<'_>,
    op: &str,
    name: &str,
    line: u32,
    col: u32,
) -> Index {
    let c = s.scalars[index];
    let w = left.wide || wide(c);
    let ty = if w { "i64" } else { "i32" };
    let right = if matches!(c, Count::One) {
        ir.value("arith.constant 1 : i32".into(), name, line, col);
        ir.value(format!("arith.constant 1 : {ty}"), name, line, col)
    } else {
        let v = format!("%{}", NAMES[index]);
        if w && !wide(c) {
            ir.value(format!("arith.extsi {v} : i32 to i64"), name, line, col)
        } else {
            v
        }
    };
    let left = if w && !left.wide {
        ir.value(
            format!("arith.extsi {} : i32 to i64", left.v),
            name,
            line,
            col,
        )
    } else {
        left.v.clone()
    };
    if !w && matches!(op, "muli" | "addi") {
        ir.overflow_temporaries(op, &left, &right, false, name, line, col);
    }
    Index {
        v: ir.value(
            format!("arith.{op} {left}, {right} : {ty}"),
            name,
            line,
            col,
        ),
        wide: w,
    }
}
fn bounds(ir: &mut Ir, s: &Specialization<'_>, line: u32) -> [String; 3] {
    let zero = ir.value("arith.constant 0 : i32".into(), "", line, 27);
    let d = if matches!(s.scalars[0], Count::One) {
        ir.value("arith.constant 1 : i32".into(), "", line, 27)
    } else {
        "%d".into()
    };
    let block = ir.value(
        format!("arith.constant {} : i32", s.block_size),
        "",
        line,
        27,
    );
    let ty = if wide(s.scalars[0]) { "i64" } else { "i32" };
    let values = [(zero, "i32"), (d, ty), (block, "i32")].map(|(v, from)| {
        ir.value(
            format!(
                "arith.{} {v} : {from} to {ty}",
                if from == ty { "bitcast" } else { "extsi" }
            ),
            "",
            line,
            27,
        )
    });
    ir.value(format!("ub.poison : {ty}"), "", line, 27);
    values
}
fn columns(ir: &mut Ir, s: &Specialization<'_>, off: &str, line: u32) -> (Index, String) {
    let w = wide(s.scalars[0]);
    let ty = if w { "i64" } else { "i32" };
    let range = ir.value(
        format!(
            "tt.make_range {{end = {} : i32, start = 0 : i32}} : {}",
            s.block_size,
            ir.vector("i32")
        ),
        "cols",
        line,
        34,
    );
    let range = if w {
        ir.value(
            format!(
                "arith.extsi {range} : {} to {}",
                ir.vector("i32"),
                ir.vector("i64")
            ),
            "cols",
            line,
            21,
        )
    } else {
        range
    };
    let off = ir.splat(off, ty, "cols", line, 21);
    if !w {
        ir.overflow_temporaries("addi", &off, &range, true, "cols", line, 21);
    }
    let cols = ir.value(
        format!("arith.addi {off}, {range} : {}", ir.vector(ty)),
        "cols",
        line,
        21,
    );
    let d = if matches!(s.scalars[0], Count::One) {
        ir.value("arith.constant 1 : i32".into(), "mask", line + 1, 22);
        ir.value(
            format!("arith.constant dense<1> : {}", ir.vector("i32")),
            "mask",
            line + 1,
            22,
        )
    } else {
        ir.splat("%d", ty, "mask", line + 1, 22)
    };
    let mask = ir.value(
        format!("arith.cmpi slt, {cols}, {d} : {}", ir.vector(ty)),
        "mask",
        line + 1,
        22,
    );
    (Index { v: cols, wide: w }, mask)
}
#[allow(clippy::too_many_arguments)]
fn load(
    ir: &mut Ir,
    pointer: &str,
    base: &Index,
    cols: &Index,
    mask: &str,
    kind: Element,
    name: &str,
    line: u32,
    base_col: u32,
    index_col: u32,
    load_col: u32,
    cast_col: u32,
) -> String {
    let pt = format!("!tt.ptr<{}>", kind.ir());
    let p = ir.value(
        format!("tt.addptr {pointer}, {} : {pt}, {}", base.v, base.ty()),
        name,
        line,
        base_col,
    );
    let p = ir.splat(&p, &pt, name, line, index_col);
    let p = ir.value(
        format!(
            "tt.addptr {p}, {} : {}, {}",
            cols.v,
            ir.vector(&pt),
            ir.vector(cols.ty())
        ),
        name,
        line,
        index_col,
    );
    ir.value(
        "arith.constant 0.000000e+00 : f32".into(),
        name,
        line,
        load_col,
    );
    let z = ir.value(
        format!("arith.constant dense<0.000000e+00> : {}", ir.vector("f32")),
        name,
        line,
        load_col,
    );
    let z = ir.cast(&z, Element::F32, kind, true, name, line, load_col);
    let v = ir.value(
        format!("tt.load {p}, {mask}, {z} : {}", ir.vector(&pt)),
        name,
        line,
        load_col,
    );
    ir.cast(&v, kind, Element::F32, true, name, line, cast_col)
}
fn div_d(ir: &mut Ir, s: &Specialization<'_>, v: &str, name: &str, line: u32, col: u32) -> String {
    let d = if matches!(s.scalars[0], Count::One) {
        ir.value("arith.constant 1 : i32".into(), name, line, col);
        ir.value("arith.constant 1.000000e+00 : f32".into(), name, line, col)
    } else {
        ir.value(
            format!(
                "arith.sitofp %d : {} to f32",
                if wide(s.scalars[0]) { "i64" } else { "i32" }
            ),
            name,
            line,
            col,
        )
    };
    ir.value(format!("arith.divf {v}, {d} : f32"), name, line, col)
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    source_ttir_scalar(s, super::scalar::ScalarType::F32)
}
pub fn source_ttir_scalar(s: &Specialization<'_>, epsilon: super::scalar::ScalarType) -> String {
    let path = serde_json::to_string(s.source_path).unwrap();
    let stdpath = serde_json::to_string(s.standard_source_path).unwrap();
    let mut ir = Ir {
        body: String::new(),
        next: 0,
        block: s.block_size,
    };
    let mut args = vec![];
    for (i, (name, k)) in [
        ("x_ptr", s.input),
        ("scale_ptr", s.scale),
        ("shift_ptr", s.shift),
        ("y_ptr", s.input),
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
            "%{name}: !tt.ptr<{}>{attr} loc(\"{name}\"({path}:12:0))",
            k.ir()
        ));
    }
    for (name, c) in NAMES.into_iter().zip(s.scalars) {
        if matches!(c, Count::One) {
            continue;
        }
        let attr = if matches!(
            c,
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
            "%{name}: {}{attr} loc(\"{name}\"({path}:12:0))",
            if wide(c) { "i64" } else { "i32" }
        ));
    }
    if !matches!(epsilon, super::scalar::ScalarType::One) {
        args.push(format!(
            "%eps: {}{} loc(\"eps\"({path}:12:0))",
            epsilon.ty().ir(),
            if epsilon.divisible() {
                " {tt.divisibility = 16 : i32}"
            } else {
                ""
            }
        ));
    }
    let row = Index {
        v: ir.value("tt.get_program_id x : i32".into(), "row", 22, 24),
        wide: false,
    };
    let xb = bin_arg(&mut ir, &row, 3, s, "muli", "x_base", 23, 20);
    let sg = bin_arg(&mut ir, &row, 1, s, "divsi", "s_base", 26, 22);
    let sb = bin_arg(&mut ir, &sg, 4, s, "muli", "s_base", 26, 37);
    let hg = bin_arg(&mut ir, &row, 2, s, "divsi", "sh_base", 27, 22);
    let hb = bin_arg(&mut ir, &hg, 5, s, "muli", "sh_base", 27, 37);
    let yb = bin_arg(&mut ir, &row, 6, s, "muli", "y_base", 28, 20);
    let vec = ir.vector("f32");
    let zeros = format!("triton.language.standard.zeros__Tc{}T_cfp32", s.block_size);
    let sumfn = format!(
        "triton.language.standard.sum__fp32S{}S_cNone_cFalse_cNone",
        s.block_size
    );
    let sumzero = ir.value(
        format!("tt.call @{zeros}() : () -> {vec}"),
        "sum_acc",
        34,
        25,
    );
    let sqzero = ir.value(
        format!("tt.call @{zeros}() : () -> {vec}"),
        "sumsq_acc",
        35,
        25,
    );
    let b = bounds(&mut ir, s, 36);
    let ty = if wide(s.scalars[0]) { "i64" } else { "i32" };
    let (carry, operands, types, results) = if s.subtract_mean {
        (
            format!(
                "%sum_in: {vec} loc(\"sum_acc\"(\"sum_acc\"({path}:34:25))), %sq_in: {vec} loc(\"sumsq_acc\"(\"sumsq_acc\"({path}:35:25)))"
            ),
            format!(", {sumzero}, {sqzero}"),
            format!(", {vec}, {vec}"),
            format!("({vec}, {vec})"),
        )
    } else {
        (
            format!("%sq_in: {vec} loc(\"sumsq_acc\"(\"sumsq_acc\"({path}:35:25)))"),
            format!(", {sqzero}"),
            format!(", {vec}"),
            vec.clone(),
        )
    };
    let lhs = if s.subtract_mean { "%acc:2" } else { "%acc" };
    writeln!(ir.body,"    {lhs} = \"scf.for\"({}, {}, {}{operands}) ({{\n    ^bb0(%off0: {ty} loc(\"off\"({path}:36:27)), {carry}):",b[0],b[1],b[2]).unwrap();
    let (cols, mask) = columns(&mut ir, s, "%off0", 37);
    let x = load(
        &mut ir, "%x_ptr", &xb, &cols, &mask, s.input, "x", 39, 28, 37, 20, 68,
    );
    let sum = if s.subtract_mean {
        Some(ir.value(
            format!("arith.addf %sum_in, {x} : {vec}"),
            "sum_acc",
            41,
            23,
        ))
    } else {
        None
    };
    let square = ir.value(format!("arith.mulf {x}, {x} : {vec}"), "sumsq_acc", 42, 25);
    let square = ir.value(
        format!("arith.addf %sq_in, {square} : {vec}"),
        "sumsq_acc",
        42,
        21,
    );
    ir.line(
        if let Some(sum) = sum {
            format!("scf.yield {sum}, {square} : {vec}, {vec}")
        } else {
            format!("scf.yield {square} : {vec}")
        },
        "",
        42,
        8,
    );
    let looploc = if s.subtract_mean {
        format!("\"sumsq_acc\"(\"sum_acc\"({path}:36:27))")
    } else {
        format!("\"sumsq_acc\"({path}:36:27)")
    };
    writeln!(
        ir.body,
        "    }}) : ({ty}, {ty}, {ty}{types}) -> {results} loc({looploc})"
    )
    .unwrap();
    let (mean, rstd) = if s.subtract_mean {
        let m = ir.value(
            format!("tt.call @{sumfn}(%acc#0) : ({vec}) -> f32"),
            "mean",
            44,
            22,
        );
        let m = div_d(&mut ir, s, &m, "mean", 44, 33);
        let v = ir.value(
            format!("tt.call @{sumfn}(%acc#1) : ({vec}) -> f32"),
            "var",
            47,
            22,
        );
        let v = div_d(&mut ir, s, &v, "var", 47, 35);
        let mm = ir.value(format!("arith.mulf {m}, {m} : f32"), "var", 47, 46);
        let v = ir.value(format!("arith.subf {v}, {mm} : f32"), "var", 47, 39);
        let zero = ir.value("arith.constant 0.000000e+00 : f32".into(), "rstd", 48, 40);
        let v = ir.value(format!("arith.maxnumf {v}, {zero} : f32"), "rstd", 48, 40);
        let eps = numeric_epsilon(&mut ir, epsilon, 48, 47);
        let v = ir.value(format!("arith.addf {v}, {eps} : f32"), "rstd", 48, 47);
        (m, ir.value(format!("math.rsqrt {v} : f32"), "rstd", 48, 24))
    } else {
        let m = ir.value("arith.constant 0.000000e+00 : f32".into(), "mean", 50, 15);
        let v = ir.value(
            format!("tt.call @{sumfn}(%acc) : ({vec}) -> f32"),
            "rstd",
            51,
            31,
        );
        let v = div_d(&mut ir, s, &v, "rstd", 51, 44);
        let eps = numeric_epsilon(&mut ir, epsilon, 51, 48);
        let v = ir.value(format!("arith.addf {v}, {eps} : f32"), "rstd", 51, 48);
        (m, ir.value(format!("math.rsqrt {v} : f32"), "rstd", 51, 24))
    };
    let b = bounds(&mut ir, s, 54);
    writeln!(
        ir.body,
        "    \"scf.for\"({}, {}, {}) ({{\n    ^bb0(%off1: {ty} loc(\"off\"({path}:54:27))):",
        b[0], b[1], b[2]
    )
    .unwrap();
    let (cols, mask) = columns(&mut ir, s, "%off1", 55);
    let x = load(
        &mut ir, "%x_ptr", &xb, &cols, &mask, s.input, "x", 57, 29, 38, 21, 69,
    );
    let sc = load(
        &mut ir,
        "%scale_ptr",
        &sb,
        &cols,
        &mask,
        s.scale,
        "sc",
        58,
        33,
        42,
        21,
        73,
    );
    let sh = load(
        &mut ir,
        "%shift_ptr",
        &hb,
        &cols,
        &mask,
        s.shift,
        "sh",
        59,
        33,
        43,
        21,
        74,
    );
    let m = if s.subtract_mean {
        ir.splat(&mean, "f32", "out", 60, 19)
    } else {
        ir.value(
            format!("arith.constant dense<0.000000e+00> : {vec}"),
            "out",
            60,
            19,
        )
    };
    let out = ir.value(format!("arith.subf {x}, {m} : {vec}"), "out", 60, 19);
    let r = ir.splat(&rstd, "f32", "out", 60, 27);
    let out = ir.value(format!("arith.mulf {out}, {r} : {vec}"), "out", 60, 27);
    ir.value("arith.constant 1.000000e+00 : f32".into(), "out", 60, 41);
    ir.value("arith.constant 1.000000e+00 : f32".into(), "out", 60, 41);
    let one = ir.value(
        format!("arith.constant dense<1.000000e+00> : {vec}"),
        "out",
        60,
        41,
    );
    let modulation = ir.value(format!("arith.addf {one}, {sc} : {vec}"), "out", 60, 41);
    let out = ir.value(
        format!("arith.mulf {out}, {modulation} : {vec}"),
        "out",
        60,
        35,
    );
    let out = ir.value(format!("arith.addf {out}, {sh} : {vec}"), "out", 60, 47);
    let pt = format!("!tt.ptr<{}>", s.input.ir());
    let yp = ir.value(
        format!("tt.addptr %y_ptr, {} : {pt}, {}", yb.v, yb.ty()),
        "",
        61,
        25,
    );
    let yp = ir.splat(&yp, &pt, "", 61, 34);
    let yp = ir.value(
        format!(
            "tt.addptr {yp}, {} : {}, {}",
            cols.v,
            ir.vector(&pt),
            ir.vector(cols.ty())
        ),
        "",
        61,
        34,
    );
    let out = ir.cast(&out, Element::F32, s.input, true, "", 61, 47);
    ir.line(
        format!("tt.store {yp}, {out}, {mask} : {}", ir.vector(&pt)),
        "",
        61,
        40,
    );
    ir.line("scf.yield".into(), "", 54, 27);
    writeln!(
        ir.body,
        "    }}) : ({ty}, {ty}, {ty}) -> () loc({path}:54:27)"
    )
    .unwrap();
    ir.line("tt.return".into(), "", 54, 4);
    let helpers = format!(
        r#"  tt.func private @{zeros}() -> {vec} attributes {{noinline = false}} {{
    %zero = arith.constant 0.000000e+00 : f32 loc({stdpath}:129:31)
    %zeros = arith.constant dense<0.000000e+00> : {vec} loc({stdpath}:129:31)
    tt.return %zeros : {vec} loc({stdpath}:129:11)
  ^bb1:
    %poison = ub.poison : {vec} loc({stdpath}:129:4)
    tt.return %poison : {vec} loc({stdpath}:129:4)
  }} loc({stdpath}:120:0)
  tt.func private @{sumfn}(%input: {vec} loc("input"({stdpath}:287:0))) -> f32 attributes {{noinline = false}} {{
    %flat = tt.reshape %input allow_reorder : {vec} -> {vec} loc({stdpath}:293:36)
    %reduced = "tt.reduce"(%flat) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %sum = tt.call @triton.language.standard._sum_combine__fp32_fp32(%a, %b) : (f32, f32) -> f32 loc({stdpath}:293:36)
      tt.reduce.return %sum : f32 loc({stdpath}:293:36)
    }}) : ({vec}) -> f32 loc({stdpath}:293:36)
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
    format!(
        "module {{\n  tt.func public @_adaln_fwd_kernel({}) attributes {{noinline = false}} {{\n{}  }} loc({path}:12:0)\n{helpers}}} loc({path}:12:0)\n",
        args.join(", "),
        ir.body.replace("\"@SOURCE@\"", &path)
    )
}

fn numeric_epsilon(
    ir: &mut Ir,
    epsilon: super::scalar::ScalarType,
    line: u32,
    column: u32,
) -> String {
    use super::scalar::ScalarType;
    match epsilon {
        ScalarType::F32 => "%eps".into(),
        ScalarType::I32 { .. }
        | ScalarType::I64 { .. }
        | ScalarType::U64 { .. }
        | ScalarType::Bool => ir.value(
            format!(
                "arith.{} %eps : {} to f32",
                if matches!(epsilon, ScalarType::Bool | ScalarType::U64 { .. }) {
                    "uitofp"
                } else {
                    "sitofp"
                },
                epsilon.ty().ir()
            ),
            "rstd",
            line,
            column,
        ),
        ScalarType::One => {
            ir.value("arith.constant 1 : i32".into(), "rstd", line, column);
            ir.value(
                "arith.constant 1.000000e+00 : f32".into(),
                "rstd",
                line,
                column,
            )
        }
        ScalarType::F64 => {
            unreachable!("Original Python numeric JIT values specialize as FP32, not FP64")
        }
    }
}
