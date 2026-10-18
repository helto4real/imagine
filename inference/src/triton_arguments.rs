//! Source JIT argument metadata, including structured values rejected by math.
use crate::triton_frontend::scalar::ScalarType;
#[derive(Clone)]
pub enum ArgumentType {
    Numeric(ScalarType),
    None,
    Pointer {
        pointee: &'static str,
        divisible_by_16: bool,
    },
    Tuple(Vec<ArgumentType>),
}
impl ArgumentType {
    pub(crate) fn signature_repr(&self) -> String {
        match self {
            Self::Numeric(kind) => format!("'{}'", kind.signature()),
            Self::None => "'constexpr'".into(),
            Self::Pointer { pointee, .. } => format!("'*{pointee}'"),
            Self::Tuple(values) => format!(
                "({}{})",
                values
                    .iter()
                    .map(Self::signature_repr)
                    .collect::<Vec<_>>()
                    .join(", "),
                if values.len() == 1 { "," } else { "" }
            ),
        }
    }
    pub(crate) fn constants(&self) -> Vec<String> {
        match self {
            Self::Numeric(ScalarType::One) => vec!["1".into()],
            Self::None => vec!["None".into()],
            Self::Tuple(values) => values.iter().flat_map(Self::constants).collect(),
            _ => vec![],
        }
    }
    pub(crate) fn attributes(&self, path: &[usize]) -> Vec<String> {
        match self {
            Self::Numeric(kind) if kind.divisible() => {
                vec![format!("{}: [['tt.divisibility', 16]]", path_repr(path))]
            }
            Self::Pointer {
                divisible_by_16: true,
                ..
            } => vec![format!("{}: [['tt.divisibility', 16]]", path_repr(path))],
            Self::Tuple(values) => values
                .iter()
                .enumerate()
                .flat_map(|(i, value)| {
                    let mut child = path.to_vec();
                    child.push(i);
                    value.attributes(&child)
                })
                .collect(),
            _ => vec![],
        }
    }
    pub(crate) fn unsupported_cause(&self) -> Option<String> {
        match self {
            Self::Numeric(_) => None,
            Self::None => {
                Some("AttributeError(\"'NoneType' object has no attribute 'type'\")".into())
            }
            Self::Tuple(_) => {
                Some("AttributeError(\"'tuple_type' object has no attribute 'scalar'\")".into())
            }
            Self::Pointer { pointee, .. } => {
                let name = match *pointee {
                    "u1" => "int1",
                    "i8" => "int8",
                    "i16" => "int16",
                    "i32" => "int32",
                    "i64" => "int64",
                    "u8" => "uint8",
                    "u16" => "uint16",
                    "u32" => "uint32",
                    "u64" => "uint64",
                    name => name,
                };
                Some(format!(
                    "IncompatibleTypeErrorImpl('invalid operands of type pointer<{name}> and triton.language.float32')"
                ))
            }
        }
    }
}
pub(crate) fn path_repr(path: &[usize]) -> String {
    format!(
        "({}{})",
        path.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if path.len() == 1 { "," } else { "" }
    )
}

/// Original constexpr source text and separate Python specialization equality.
pub struct ConstexprKey {
    pub source: String,
    pub memory: String,
}
