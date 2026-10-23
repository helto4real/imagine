//! Source-owned torch.dtype singletons for every locked ATen scalar code.
//! Shell/bit/unsigned types retain raw codes even when tch::Kind cannot express
//! them. This module supplies descriptors, not a reordered default registry.
use crate::{
    execution_value::Value,
    native::Error,
    patching::options::{MetadataObject, MetadataValue, Options, SourceNumber, SourceText},
};
use std::any::Any;
use tch::Kind;
#[path = "dtype_objects_data.rs"]
mod data;
pub struct SourceDtype {
    code: usize,
}
thread_local! {static DTYPES:Vec<Options>=(0..data::DTYPES.len()).map(|code|MetadataObject::wrap(SourceDtype {code})).collect();}
fn error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "RuntimeError",
        message: message.into(),
    }
}
impl SourceDtype {
    /// Descriptor lookup is not Python's nonconstructible torch.dtype class.
    pub fn from_scalar_type(code: i32) -> Option<Options> {
        usize::try_from(code)
            .ok()
            .filter(|code| *code < data::DTYPES.len())
            .map(|code| DTYPES.with(|values| values[code].clone()))
    }
    pub fn lookup(name: &str) -> Option<Options> {
        let name = name.strip_prefix("torch.").unwrap_or(name);
        data::DTYPES
            .iter()
            .position(|(primary, alias, _, _)| {
                *primary == name || !alias.is_empty() && *alias == name
            })
            .and_then(|code| Self::from_scalar_type(code as i32))
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn scalar_type(&self) -> i32 {
        self.code as i32
    }
    pub fn name(&self) -> &'static str {
        data::DTYPES[self.code].0
    }
    pub fn representation(&self) -> String {
        format!("torch.{}", self.name())
    }
    pub fn itemsize(&self) -> usize {
        data::DTYPES[self.code].3
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn hash(&self) -> i64 {
        crate::python_hash::identity_hash_address(self.identity_token() as usize)
    }
    pub fn is_floating_point(&self) -> bool {
        matches!(self.code,5..=7|15|23..=26|44..=45)
    }
    pub fn is_complex(&self) -> bool {
        matches!(self.code, 8..=10)
    }
    pub fn is_signed(&self) -> Result<bool, Error> {
        match self.code {
            12..=14 | 16..=17 => Err(error("isSignedType not supported for quantized types")),
            18..=22 => Err(error("Bits types are undefined")),
            0 | 11 | 27..=36 | 44 => Ok(false),
            _ => Ok(true),
        }
    }
    pub fn abbreviation(&self) -> &'static str {
        match self.code {
            0 => "u8",
            1 => "i8",
            2 => "i16",
            3 => "i32",
            4 => "i64",
            5 => "f16",
            6 => "f32",
            7 => "f64",
            8 => "c32",
            9 => "c64",
            10 => "c128",
            11 => "b8",
            15 => "bf16",
            18 => "b1x8",
            19 => "b2x4",
            20 => "b4x2",
            21 => "b8x1",
            22 => "b16x1",
            23 => "f8e5m2",
            24 => "f8e4m3fn",
            25 => "f8e5m2fnuz",
            26 => "f8e4m3fnuz",
            27 => "u16",
            28 => "u32",
            29 => "u64",
            44 => "f8e8m0fnu",
            45 => "f4e2m1fnx2",
            _ => self.name(),
        }
    }
    pub fn property(&self, name: &str) -> Result<Options, Error> {
        Ok(match name {
            "itemsize" => SourceNumber::from_i64(self.itemsize() as i64),
            "is_floating_point" => SourceNumber::boolean(self.is_floating_point()),
            "is_complex" => SourceNumber::boolean(self.is_complex()),
            "is_signed" => SourceNumber::boolean(self.is_signed()?),
            "abbr" => SourceText::from_utf8(self.abbreviation()),
            _ => {
                return Err(Error::BindingGap {
                    backend: "torch_dtype".into(),
                    operation: "other attributes require the source object's attribute protocol"
                        .into(),
                });
            }
        })
    }
    pub fn reduce_name(&self) -> Options {
        SourceText::from_utf8(self.name())
    }
    pub fn to_real(&self) -> Options {
        Self::from_scalar_type(match self.code {
            8 => 5,
            9 => 6,
            10 => 7,
            _ => self.scalar_type(),
        })
        .unwrap()
    }
    pub fn to_complex(&self) -> Result<Options, Error> {
        let code = match self.code {
            5 => 8,
            6 | 15 => 9,
            7 => 10,
            8..=10 => self.scalar_type(),
            _ => {
                return Err(error(format!(
                    "Unknown Complex ScalarType for {}",
                    data::DTYPES[self.code].2
                )));
            }
        };
        Ok(Self::from_scalar_type(code).unwrap())
    }
    pub fn kind(&self) -> Option<Kind> {
        Some(match self.code {
            0 => Kind::Uint8,
            1 => Kind::Int8,
            2 => Kind::Int16,
            3 => Kind::Int,
            4 => Kind::Int64,
            5 => Kind::Half,
            6 => Kind::Float,
            7 => Kind::Double,
            8 => Kind::ComplexHalf,
            9 => Kind::ComplexFloat,
            10 => Kind::ComplexDouble,
            11 => Kind::Bool,
            12 => Kind::QInt8,
            13 => Kind::QUInt8,
            14 => Kind::QInt32,
            15 => Kind::BFloat16,
            23 => Kind::Float8e5m2,
            24 => Kind::Float8e4m3fn,
            25 => Kind::Float8e5m2fnuz,
            26 => Kind::Float8e4m3fnuz,
            _ => return None,
        })
    }
}
impl MetadataValue for SourceDtype {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        self.kind()
            .map(Value::DType)
            .ok_or_else(|| Error::BindingGap {
                backend: "torch_dtype".into(),
                operation: "extended dtype requires its raw ATen scalar-code consumer".into(),
            })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| std::ptr::eq(self, other))
    }
    fn type_name(&self) -> &str {
        "torch.dtype"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(true)
    }
}
