//! Shared typed scalar/pointer IR mechanics for the frozen kernel programs.
use super::Element;
use std::fmt::Write;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Ty {
    I1,
    I8,
    CharPointer,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F64,
    BytePointer,
    Float(Element),
    Pointer(Element),
}
impl Ty {
    pub(super) fn ir(self) -> String {
        match self {
            Self::I1 => "i1".into(),
            Self::I8 => "i8".into(),
            Self::CharPointer => "!tt.ptr<i8>".into(),
            Self::I32 => "i32".into(),
            Self::I64 => "i64".into(),
            Self::U8 => "i8".into(),
            Self::U16 => "i16".into(),
            Self::U32 => "i32".into(),
            Self::U64 => "i64".into(),
            Self::F64 => "f64".into(),
            Self::BytePointer => "!tt.ptr<i8>".into(),
            Self::Float(k) => k.ir().into(),
            Self::Pointer(k) => format!("!tt.ptr<{}>", k.ir()),
        }
    }
    pub(super) fn mangle(self) -> String {
        match self {
            Self::I1 => "u1".into(),
            Self::I8 => "i8".into(),
            Self::CharPointer => "Pi8".into(),
            Self::I32 => "i32".into(),
            Self::I64 => "i64".into(),
            Self::U8 => "u8".into(),
            Self::U16 => "u16".into(),
            Self::U32 => "u32".into(),
            Self::U64 => "u64".into(),
            Self::F64 => "fp64".into(),
            Self::BytePointer => "Pu8".into(),
            Self::Float(k) => match k {
                Element::F64 => "fp64",
                Element::F32 => "fp32",
                Element::F16 => "fp16",
                Element::BF16 => "bf16",
                Element::E4M3 => "fp8e4nv",
                Element::E5M2 => "fp8e5",
            }
            .into(),
            Self::Pointer(k) => format!("P{}", Self::Float(k).mangle()),
        }
    }
}
#[derive(Clone)]
pub(super) struct Value {
    pub(super) id: String,
    pub(super) ty: Ty,
    pub(super) shape: Vec<u32>,
    pub(super) literal: Option<i64>,
    pub(super) float_literal: Option<u64>,
    pub(super) constexpr: bool,
}
impl Value {
    pub(super) fn integer(n: i64) -> Self {
        Self {
            id: String::new(),
            ty: if i32::try_from(n).is_ok() {
                Ty::I32
            } else if u32::try_from(n).is_ok() {
                Ty::U32
            } else {
                Ty::I64
            },
            shape: vec![],
            literal: Some(n),
            float_literal: None,
            constexpr: true,
        }
    }
    pub(super) fn number(value: f64) -> Self {
        let absolute = value.abs();
        let ty = if value == 0.0
            || !value.is_finite()
            || (f64::from(f32::MIN_POSITIVE) <= absolute && absolute <= f64::from(f32::MAX))
        {
            Ty::Float(Element::F32)
        } else {
            Ty::F64
        };
        Self {
            id: String::new(),
            ty,
            shape: vec![],
            literal: None,
            float_literal: Some(value.to_bits()),
            constexpr: true,
        }
    }
    pub(super) fn is_block(&self) -> bool {
        !self.shape.is_empty()
    }
    pub(super) fn typed(&self, _block: u32) -> String {
        self.typed_shape(&self.shape)
    }
    pub(super) fn typed_shape(&self, shape: &[u32]) -> String {
        if shape.is_empty() {
            self.ty.ir()
        } else {
            format!(
                "tensor<{}x{}>",
                shape
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
                self.ty.ir()
            )
        }
    }
}
pub(super) struct Builder {
    pub(super) body: String,
    pub(super) next: usize,
    pub(super) block: u32,
    pub(super) path: String,
    pub(super) helpers: Vec<(String, String)>,
    pub(super) compute: Element,
}
impl Ty {
    fn floating(self) -> bool {
        matches!(self, Self::Float(_) | Self::F64)
    }
    fn pointer(self) -> bool {
        matches!(
            self,
            Self::Pointer(_) | Self::BytePointer | Self::CharPointer
        )
    }
    fn kind(self) -> u8 {
        if self.pointer() {
            3
        } else if self.floating() {
            2
        } else if self == Self::I1 {
            0
        } else {
            1
        }
    }
    pub(super) fn pointee(self) -> Self {
        match self {
            Self::Pointer(k) => Self::Float(k),
            Self::BytePointer => Self::U8,
            Self::CharPointer => Self::I8,
            _ => panic!("Original pointer type"),
        }
    }
}
impl Builder {
    pub(super) fn loc(&self, name: &str, p: (u32, u32)) -> String {
        let pos = format!("{}:{}:{}", self.path, p.0, p.1);
        if name.is_empty() {
            format!("loc({pos})")
        } else {
            format!("loc(\"{name}\"({pos}))")
        }
    }
    pub(super) fn line(&mut self, op: String, name: &str, p: (u32, u32)) {
        writeln!(self.body, "    {op} {}", self.loc(name, p)).unwrap();
    }
    pub(super) fn emit(
        &mut self,
        op: String,
        ty: Ty,
        vector: bool,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let shape = if vector { vec![self.block] } else { vec![] };
        self.emit_shape(op, ty, &shape, name, p)
    }
    pub(super) fn emit_shape(
        &mut self,
        op: String,
        ty: Ty,
        shape: &[u32],
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let id = format!("%v{}", self.next);
        self.next += 1;
        self.line(format!("{id} = {op}"), name, p);
        Value {
            id,
            ty,
            shape: shape.to_vec(),
            literal: None,
            float_literal: None,
            constexpr: false,
        }
    }
    pub(super) fn constant(
        &mut self,
        n: i64,
        ty: Ty,
        vector: bool,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let shape = if vector { vec![self.block] } else { vec![] };
        self.materialize(&Value::integer(n), ty, &shape, name, p)
    }
    pub(super) fn float_constant(
        &mut self,
        value: f64,
        ty: Ty,
        shape: &[u32],
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let (bits, width) = match ty {
            Ty::F64 | Ty::Float(Element::F64) => (value.to_bits(), 16),
            Ty::Float(Element::F32) => (ieee_narrow(value, 8, 23, 127), 8),
            Ty::Float(Element::F16) => (ieee_narrow(value, 5, 10, 15), 4),
            Ty::Float(Element::BF16) => (ieee_narrow(value, 8, 7, 127), 4),
            _ => panic!("Original scalar floating constant type"),
        };
        let text = format!("0x{bits:0width$X}");
        let text = if shape.is_empty() {
            text
        } else {
            format!("dense<{text}>")
        };
        let template = Value {
            id: String::new(),
            ty,
            shape: shape.to_vec(),
            literal: None,
            float_literal: None,
            constexpr: false,
        };
        let mut out = self.emit_shape(
            format!("arith.constant {text} : {}", template.typed(self.block)),
            ty,
            shape,
            name,
            p,
        );
        out.float_literal = Some(value.to_bits());
        out
    }
    pub(super) fn materialize(
        &mut self,
        value: &Value,
        ty: Ty,
        shape: &[u32],
        name: &str,
        p: (u32, u32),
    ) -> Value {
        if let Some(bits) = value.float_literal {
            return self.float_constant(f64::from_bits(bits), ty, shape, name, p);
        }
        let n = value.literal.expect("Original scalar literal");
        if ty.floating() {
            return self.float_constant(n as f64, ty, shape, name, p);
        }
        let text = if ty == Ty::I1 {
            if n == 0 {
                "false".into()
            } else {
                "true".into()
            }
        } else {
            n.to_string()
        };
        let text = if shape.is_empty() {
            text
        } else {
            format!("dense<{text}>")
        };
        let template = Value {
            id: String::new(),
            ty,
            shape: shape.to_vec(),
            literal: None,
            float_literal: None,
            constexpr: false,
        };
        let suffix = if ty == Ty::I1 && shape.is_empty() {
            String::new()
        } else {
            format!(" : {}", template.typed(self.block))
        };
        let mut out = self.emit_shape(format!("arith.constant {text}{suffix}"), ty, shape, name, p);
        out.literal = Some(n);
        out
    }
    pub(super) fn tensor(&mut self, value: Value, name: &str, p: (u32, u32)) -> Value {
        if value.constexpr {
            self.materialize(&value, value.ty, &[], name, p)
        } else {
            value
        }
    }
    pub(super) fn expand(&mut self, value: Value, axis: usize, name: &str, p: (u32, u32)) -> Value {
        let mut shape = value.shape.clone();
        shape.insert(axis, 1);
        self.emit_shape(
            format!(
                "tt.expand_dims {} {{axis = {axis} : i32}} : {} -> {}",
                value.id,
                value.typed(self.block),
                value.typed_shape(&shape)
            ),
            value.ty,
            &shape,
            name,
            p,
        )
    }
    pub(super) fn broadcast_to(
        &mut self,
        mut value: Value,
        shape: &[u32],
        name: &str,
        p: (u32, u32),
    ) -> Value {
        if value.shape == shape {
            return value;
        }
        if !value.is_block() {
            if value.literal.is_some() || value.float_literal.is_some() {
                return self.materialize(&value, value.ty, shape, name, p);
            }
            return self.emit_shape(
                format!(
                    "tt.splat {} : {} -> {}",
                    value.id,
                    value.ty.ir(),
                    value.typed_shape(shape)
                ),
                value.ty,
                shape,
                name,
                p,
            );
        }
        while value.shape.len() < shape.len() {
            value = self.expand(value, 0, name, p);
        }
        if value.shape == shape {
            return value;
        }
        self.emit_shape(
            format!(
                "tt.broadcast {} : {} -> {}",
                value.id,
                value.typed(self.block),
                value.typed_shape(shape)
            ),
            value.ty,
            shape,
            name,
            p,
        )
    }
    pub(super) fn broadcast_pair(
        &mut self,
        a: Value,
        b: Value,
        name: &str,
        p: (u32, u32),
    ) -> (Value, Value) {
        let rank = a.shape.len().max(b.shape.len());
        let mut shape = vec![1; rank];
        for value in [&a, &b] {
            for (i, &d) in value.shape.iter().rev().enumerate() {
                let target = &mut shape[rank - i - 1];
                assert!(
                    *target == 1 || d == 1 || *target == d,
                    "Original compatible block shape"
                );
                *target = (*target).max(d);
            }
        }
        (
            self.broadcast_to(a, &shape, name, p),
            self.broadcast_to(b, &shape, name, p),
        )
    }
    pub(super) fn cast(&mut self, mut value: Value, ty: Ty, name: &str, p: (u32, u32)) -> Value {
        if value.ty == ty
            || matches!(
                (value.ty, ty),
                (Ty::F64, Ty::Float(Element::F64)) | (Ty::Float(Element::F64), Ty::F64)
            )
        {
            value.ty = ty;
            return value;
        }
        let shape = value.shape.clone();
        let target = value.typed_shape(&shape).replace(&value.ty.ir(), &ty.ir());
        if matches!(value.ty,Ty::Float(k)if k.fp8()) || matches!(ty,Ty::Float(k)if k.fp8()) {
            return self.emit_shape(
                format!(
                    "tt.fp_to_fp {}{} : {} -> {target}",
                    value.id,
                    if matches!(ty,Ty::Float(k)if k.fp8()) {
                        ", rounding = rtne"
                    } else {
                        ""
                    },
                    value.typed(self.block)
                ),
                ty,
                &shape,
                name,
                p,
            );
        }
        if matches!(value.ty, Ty::Float(Element::F16 | Element::BF16))
            && ty != Ty::Float(Element::F32)
        {
            let value = self.cast(value, Ty::Float(Element::F32), name, p);
            return self.cast(value, ty, name, p);
        }
        let op = match (value.ty, ty) {
            (Ty::I32, Ty::I64 | Ty::U64) | (Ty::I8, Ty::I32 | Ty::I64 | Ty::U64) => "extsi",
            (Ty::I32 | Ty::I64 | Ty::U64, Ty::I8)
            | (Ty::I64 | Ty::U64, Ty::I32 | Ty::U32)
            | (Ty::I32 | Ty::I64 | Ty::U32 | Ty::U64, Ty::U8 | Ty::U16)
            | (Ty::U16, Ty::U8) => "trunci",
            (Ty::I32, Ty::U32) | (Ty::U32, Ty::I32) | (Ty::I64, Ty::U64) | (Ty::U64, Ty::I64) => {
                "bitcast"
            }
            (Ty::I1 | Ty::U8 | Ty::U16, Ty::I32 | Ty::I64 | Ty::U32 | Ty::U64)
            | (Ty::U32, Ty::I64 | Ty::U64) => "extui",
            (Ty::I1 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64, t) if t.floating() => "uitofp",
            (Ty::I8 | Ty::I32 | Ty::I64, t) if t.floating() => "sitofp",
            (t, Ty::I8 | Ty::I32 | Ty::I64) if t.floating() => "fptosi",
            (t, Ty::U8) if t.floating() => "fptoui",
            (Ty::Float(Element::F16 | Element::BF16), Ty::Float(Element::F32))
            | (Ty::Float(_), Ty::F64 | Ty::Float(Element::F64)) => "extf",
            (Ty::F64 | Ty::Float(Element::F64), Ty::Float(_))
            | (Ty::Float(Element::F32), Ty::Float(_)) => "truncf",
            _ => panic!("Original kernel cast"),
        };
        self.emit_shape(
            format!(
                "arith.{op} {} : {} to {target}",
                value.id,
                value.typed(self.block)
            ),
            ty,
            &shape,
            name,
            p,
        )
    }
    fn result_type(a: Ty, ac: bool, b: Ty, bc: bool, division: bool) -> Ty {
        let result = if ac && a.kind() <= b.kind() {
            b
        } else if bc && b.kind() <= a.kind() {
            a
        } else if matches!(a, Ty::F64 | Ty::Float(Element::F64))
            || matches!(b, Ty::F64 | Ty::Float(Element::F64))
        {
            Ty::F64
        } else if a == Ty::Float(Element::F32) || b == Ty::Float(Element::F32) {
            Ty::Float(Element::F32)
        } else if a == Ty::Float(Element::F16) || b == Ty::Float(Element::F16) {
            Ty::Float(Element::F16)
        } else if a == Ty::Float(Element::BF16) && b == Ty::Float(Element::BF16) {
            Ty::Float(Element::BF16)
        } else if a.floating() || b.floating() {
            Ty::Float(Element::F32)
        } else if a == Ty::U64 || b == Ty::U64 {
            Ty::U64
        } else if a == Ty::I64 || b == Ty::I64 {
            Ty::I64
        } else if a == Ty::U32 || b == Ty::U32 {
            Ty::U32
        } else if a == Ty::I32 || b == Ty::I32 {
            Ty::I32
        } else {
            a
        };
        if division && matches!(result, Ty::Float(Element::F16 | Element::BF16)) {
            Ty::Float(Element::F32)
        } else {
            result
        }
    }
    pub(super) fn bin(&mut self, op: &str, a: Value, b: Value, name: &str, p: (u32, u32)) -> Value {
        if a.constexpr && b.constexpr {
            let (x, y) = (
                a.literal.expect("Frozen integer constexpr expression"),
                b.literal.expect("Frozen integer constexpr expression"),
            );
            let n = match op {
                "add" => x + y,
                "sub" => x - y,
                "mul" => x * y,
                "div" => {
                    let q = x / y;
                    let r = x % y;
                    q - i64::from(r != 0 && (r < 0) != (y < 0))
                }
                "rem" => {
                    let r = x % y;
                    if r != 0 && (r < 0) != (y < 0) {
                        r + y
                    } else {
                        r
                    }
                }
                "eq" => i64::from(x == y),
                "lt" => i64::from(x < y),
                "and" => x & y,
                "or" => x | y,
                "shl" => x << y,
                "shr" => x >> y,
                _ => panic!("Frozen constexpr expression"),
            };
            let mut v = Value::integer(n);
            if matches!(op, "eq" | "lt") {
                v.ty = Ty::I1;
            }
            return v;
        }
        let originals = (a.clone(), b.clone());
        let comparison = matches!(op, "eq" | "lt" | "le" | "gt" | "ge");
        let (ac, bc) = (a.constexpr && !comparison, b.constexpr && !comparison);
        let mut a = self.tensor(a, name, p);
        let mut b = self.tensor(b, name, p);
        let pointer = a.ty.pointer() || b.ty.pointer();
        if !pointer {
            let ty = Self::result_type(a.ty, ac, b.ty, bc, op == "true_div");
            a = if ac {
                self.materialize(&originals.0, ty, &[], name, p)
            } else {
                self.cast(a, ty, name, p)
            };
            b = if bc {
                self.materialize(&originals.1, ty, &[], name, p)
            } else {
                self.cast(b, ty, name, p)
            };
        }
        (a, b) = self.broadcast_pair(a, b, name, p);
        if pointer {
            return self.emit_shape(
                format!(
                    "tt.addptr {}, {} : {}, {}",
                    a.id,
                    b.id,
                    a.typed(self.block),
                    b.typed(self.block)
                ),
                a.ty,
                &a.shape,
                name,
                p,
            );
        }
        let arithmetic = match (op, a.ty.floating()) {
            ("true_div", true) => "divf",
            ("add", true) => "addf",
            ("sub", true) => "subf",
            ("mul", true) => "mulf",
            ("add", false) => "addi",
            ("sub", false) => "subi",
            ("mul", false) => "muli",
            ("div", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "divui",
            ("div", false) => "divsi",
            ("rem", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "remui",
            ("rem", false) => "remsi",
            ("and", false) => "andi",
            ("or", false) => "ori",
            ("shl", false) => "shli",
            ("shr", false) => {
                if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) {
                    "shrui"
                } else {
                    "shrsi"
                }
            }
            ("eq", true) => "cmpf oeq,",
            ("lt", true) => "cmpf olt,",
            ("le", true) => "cmpf ole,",
            ("gt", true) => "cmpf ogt,",
            ("ge", true) => "cmpf oge,",
            ("eq", false) => "cmpi eq,",
            ("lt", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "cmpi ult,",
            ("lt", false) => "cmpi slt,",
            ("le", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "cmpi ule,",
            ("le", false) => "cmpi sle,",
            ("gt", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "cmpi ugt,",
            ("gt", false) => "cmpi sgt,",
            ("ge", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "cmpi uge,",
            ("ge", false) => "cmpi sge,",
            ("maximum", true) => "maxnumf",
            ("minimum", true) => "minnumf",
            ("maximum", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "maxui",
            ("maximum", false) => "maxsi",
            ("minimum", false) if matches!(a.ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64) => "minui",
            ("minimum", false) => "minsi",
            _ => panic!("Frozen binary expression"),
        };
        if matches!(op, "add" | "sub" | "mul") && a.ty == Ty::I32 {
            let x = self.cast(a.clone(), Ty::I64, name, p);
            let y = self.cast(b.clone(), Ty::I64, name, p);
            let result = self.emit_shape(
                format!(
                    "arith.{arithmetic} {}, {} : {}",
                    x.id,
                    y.id,
                    x.typed(self.block)
                ),
                Ty::I64,
                &a.shape,
                name,
                p,
            );
            let upper = self.constant(2147483647, Ty::I64, false, name, p);
            let lower = self.constant(-2147483648, Ty::I64, false, name, p);
            let upper = self.broadcast_to(upper, &a.shape, name, p);
            let up = self.emit_shape(
                format!(
                    "arith.cmpi sle, {}, {} : {}",
                    result.id,
                    upper.id,
                    result.typed(self.block)
                ),
                Ty::I1,
                &a.shape,
                name,
                p,
            );
            let lower = self.broadcast_to(lower, &a.shape, name, p);
            let low = self.emit_shape(
                format!(
                    "arith.cmpi sge, {}, {} : {}",
                    result.id,
                    lower.id,
                    result.typed(self.block)
                ),
                Ty::I1,
                &a.shape,
                name,
                p,
            );
            self.emit_shape(
                format!(
                    "arith.andi {}, {} : {}",
                    up.id,
                    low.id,
                    up.typed(self.block)
                ),
                Ty::I1,
                &a.shape,
                name,
                p,
            );
        }
        self.emit_shape(
            format!(
                "arith.{arithmetic} {}, {} : {}",
                a.id,
                b.id,
                a.typed(self.block)
            ),
            if comparison { Ty::I1 } else { a.ty },
            &a.shape,
            name,
            p,
        )
    }
    pub(super) fn extreme(
        &mut self,
        op: &str,
        a: Value,
        b: Value,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let a = self.tensor(a, name, p);
        let b = self.tensor(b, name, p);
        let a = if a.ty == Ty::Float(Element::BF16) {
            self.cast(a, Ty::Float(Element::F32), name, p)
        } else {
            a
        };
        let b = if b.ty == Ty::Float(Element::BF16) {
            self.cast(b, Ty::Float(Element::F32), name, p)
        } else {
            b
        };
        self.bin(op, a, b, name, p)
    }
    pub(super) fn select(
        &mut self,
        c: Value,
        x: Value,
        y: Value,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let c = self.tensor(c, name, p);
        let originals = (x.clone(), y.clone());
        let (xc, yc) = (x.constexpr, y.constexpr);
        let x = self.tensor(x, name, p);
        let y = self.tensor(y, name, p);
        let ty = Self::result_type(x.ty, xc, y.ty, yc, false);
        let x = if xc {
            self.materialize(&originals.0, ty, &[], name, p)
        } else {
            self.cast(x, ty, name, p)
        };
        let y = if yc {
            self.materialize(&originals.1, ty, &[], name, p)
        } else {
            self.cast(y, ty, name, p)
        };
        let (mut x, mut y) = self.broadcast_pair(x, y, name, p);
        let c = if c.is_block() {
            let (v, c) = self.broadcast_pair(x, c, name, p);
            x = v;
            y = self.broadcast_to(y, &x.shape, name, p);
            c
        } else {
            self.broadcast_to(c, &x.shape, name, p)
        };
        self.emit_shape(
            format!(
                "arith.select {}, {}, {} : {}, {}",
                c.id,
                x.id,
                y.id,
                c.typed(self.block),
                x.typed(self.block)
            ),
            ty,
            &x.shape,
            name,
            p,
        )
    }
    pub(super) fn math(&mut self, op: &str, value: Value, name: &str, p: (u32, u32)) -> Value {
        self.emit_shape(
            format!("math.{op} {} : {}", value.id, value.typed(self.block)),
            value.ty,
            &value.shape,
            name,
            p,
        )
    }
    pub(super) fn load(&mut self, pointer: Value, mask: Value, name: &str, p: (u32, u32)) -> Value {
        let zero = self.float_constant(0.0, Ty::Float(Element::F32), &[], name, p);
        let zero = self.broadcast_to(zero, &pointer.shape, name, p);
        let ty = pointer.ty.pointee();
        let zero = self.cast(zero, ty, name, p);
        self.emit_shape(
            format!(
                "tt.load {}, {}, {} : {}",
                pointer.id,
                mask.id,
                zero.id,
                pointer.typed(self.block)
            ),
            ty,
            &pointer.shape,
            name,
            p,
        )
    }
    pub(super) fn store(&mut self, pointer: Value, value: Value, mask: Value, p: (u32, u32)) {
        self.store_optional(pointer, value, Some(mask), p)
    }
    pub(super) fn store_optional(
        &mut self,
        pointer: Value,
        value: Value,
        mask: Option<Value>,
        p: (u32, u32),
    ) {
        let value = self.cast(value, pointer.ty.pointee(), "", p);
        self.line(
            format!(
                "tt.store {}, {}{} : {}",
                pointer.id,
                value.id,
                mask.map_or(String::new(), |m| format!(", {}", m.id)),
                pointer.typed(self.block)
            ),
            "",
            p,
        )
    }
}
fn ieee_narrow(value: f64, exponent_bits: u32, fraction_bits: u32, bias: i32) -> u64 {
    let bits = value.to_bits();
    let width = 1 + exponent_bits + fraction_bits;
    let sign = (bits >> 63) << (width - 1);
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let max_exp = (1u64 << exponent_bits) - 1;
    if exponent == 0x7ff {
        let payload = if fraction == 0 {
            0
        } else {
            (fraction >> (52 - fraction_bits)).max(1)
        };
        return sign | (max_exp << fraction_bits) | payload;
    }
    if exponent == 0 && fraction == 0 {
        return sign;
    }
    let unbiased = if exponent == 0 {
        -1022
    } else {
        exponent - 1023
    };
    let significand = if exponent == 0 {
        fraction
    } else {
        fraction | (1u64 << 52)
    };
    let mut target_exp = unbiased + bias;
    let shift = 52 - fraction_bits
        + if target_exp < 1 {
            (1 - target_exp) as u32
        } else {
            0
        };
    let mut rounded = if shift >= 64 {
        0
    } else {
        let integer = significand >> shift;
        let remainder = significand & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        integer + u64::from(remainder > halfway || (remainder == halfway && integer & 1 != 0))
    };
    if target_exp < 1 {
        return sign | rounded;
    }
    if rounded >= 1u64 << (fraction_bits + 1) {
        rounded >>= 1;
        target_exp += 1;
    }
    if target_exp >= max_exp as i32 {
        return sign | (max_exp << fraction_bits);
    }
    sign | ((target_exp as u64) << fraction_bits) | (rounded & ((1u64 << fraction_bits) - 1))
}

impl Builder {
    pub(super) fn range(&mut self, end: Value, name: &str, p: (u32, u32)) -> Value {
        let n = u32::try_from(end.literal.unwrap()).unwrap();
        self.emit_shape(
            format!("tt.make_range {{end = {n} : i32, start = 0 : i32}} : tensor<{n}xi32>"),
            Ty::I32,
            &[n],
            name,
            p,
        )
    }

    pub(super) fn load_legacy(
        &mut self,
        pointer: Value,
        mask: Option<Value>,
        other: Option<Value>,
        name: &str,
        p: (u32, u32),
    ) -> Value {
        let ty = pointer.ty.pointee();
        let other = other.map(|v| self.tensor(v, name, p));
        let mask = mask.map(|v| self.broadcast_to(v, &pointer.shape, name, p));
        let other = other.map(|v| self.broadcast_to(v, &pointer.shape, name, p));
        let other = other.map(|v| self.cast(v, ty, name, p));
        let mut operands = pointer.id.clone();
        if let Some(mask) = mask {
            write!(operands, ", {}", mask.id).unwrap();
            if let Some(other) = other {
                write!(operands, ", {}", other.id).unwrap();
            }
        }
        self.emit_shape(
            format!("tt.load {operands} : {}", pointer.typed(self.block)),
            ty,
            &pointer.shape,
            name,
            p,
        )
    }
}
