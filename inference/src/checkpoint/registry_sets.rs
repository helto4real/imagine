//! CPython BUILD_SET for registry descriptors with exact identity equality.
//! Controlled reference bindings may supply captured hashes under the scoped
//! D-05 registry criterion. Arbitrary user equality needs its general protocol.
use crate::{native::Error, patching::options::Options, source_set::Table};
pub struct IdentityLiteral {
    table: Table<Options>,
}
impl IdentityLiteral {
    pub fn build(
        values: Vec<Options>,
        mut hash: impl FnMut(&Options) -> Result<i64, Error>,
    ) -> Result<Self, Error> {
        let mut table = Table::default();
        for value in values {
            let hash = hash(&value)?;
            table.add(hash, value, Options::same_object);
        }
        Ok(Self { table })
    }
    pub fn iter(&self) -> impl Iterator<Item = &Options> {
        self.table.iter()
    }
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }
}
/// Source literal operands are evaluated left to right before BUILD_SET hashes
/// the first value. In particular a late getter failure causes no hash calls.
pub fn literal_from_paths(
    paths: &[&str],
    bindings: &mut dyn Bindings,
) -> Result<IdentityLiteral, Error> {
    let values = paths
        .iter()
        .map(|path| bindings.load(path))
        .collect::<Result<Vec<_>, _>>()?;
    IdentityLiteral::build(values, |value| bindings.hash(value))
}
pub trait Bindings {
    /// The actual original attribute chain, preserving each getter operation.
    fn load(&mut self, path: &str) -> Result<Options, Error>;
    /// Only actual exact identity-equality owners are admitted by this binding.
    fn hash(&mut self, value: &Options) -> Result<i64, Error>;
}
pub const NEW_DTYPES: &[&str] = &[
    "torch.float8_e5m2",
    "torch.float8_e4m3fn",
    "torch.float8_e5m2fnuz",
    "torch.float8_e4m3fnuz",
    "torch.float8_e8m0fnu",
    "torch.float4_e2m1fn_x2",
    "torch.bits8",
    "torch.bits16",
    "torch.bits1x8",
    "torch.bits2x4",
    "torch.bits4x2",
    "torch.complex32",
    "torch.uint16",
    "torch.uint32",
    "torch.uint64",
];
pub const REBUILD_FUNCTIONS: &[&str] = &[
    "torch._utils._rebuild_parameter",
    "torch._utils._rebuild_parameter_with_state",
    "torch._utils._rebuild_qtensor",
    "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_tensor_v3",
    "torch._utils._rebuild_sparse_tensor",
    "torch._utils._rebuild_meta_tensor_no_storage",
    "torch._utils._rebuild_nested_tensor",
    "torch._utils._rebuild_wrapper_subclass",
    "torch._utils._rebuild_device_tensor_from_numpy",
    "torch._utils._rebuild_device_tensor_from_cpu_tensor",
];
