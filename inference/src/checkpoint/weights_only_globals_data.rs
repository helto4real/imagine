// Generated from locked Torch source; do not edit.
// _utils.py sha256: 56b9c2263e223bd59b839b70a39923126ea297ff2fe9f47dca03b1ae3715a9d8
// _weights_only_unpickler.py sha256: d9fdcb14f255d60d440e158e006501447417db95480cdd73f66fb1880b84578b
#[rustfmt::skip]
pub(super) const IMPORTS: &[(&str,&str)] = &[
("__builtin__", "builtins"),
("copy_reg", "copyreg"),
("Queue", "queue"),
("repr", "reprlib"),
("_abcoll", "collections.abc"),
("UserDict", "collections"),
("UserList", "collections"),
("UserString", "collections"),
("whichdb", "dbm"),
("StringIO", "io"),
("cStringIO", "io"),
];
#[rustfmt::skip]
pub(super) const NAMES: &[super::NameMapping] = &[
(("__builtin__","xrange"),("builtins","range")),
(("__builtin__","reduce"),("functools","reduce")),
(("__builtin__","intern"),("sys","intern")),
(("__builtin__","unichr"),("builtins","chr")),
(("__builtin__","unicode"),("builtins","str")),
(("__builtin__","long"),("builtins","int")),
(("itertools","izip"),("builtins","zip")),
(("itertools","imap"),("builtins","map")),
(("itertools","ifilter"),("builtins","filter")),
(("itertools","ifilterfalse"),("itertools","filterfalse")),
(("itertools","izip_longest"),("itertools","zip_longest")),
(("UserDict","IterableUserDict"),("collections","UserDict")),
(("UserList","UserList"),("collections","UserList")),
(("UserString","UserString"),("collections","UserString")),
(("__builtin__","basestring"),("builtins","str")),
(("exceptions","StandardError"),("builtins","Exception")),
(("UserDict","UserDict"),("collections","UserDict")),
];
#[rustfmt::skip]
pub(super) const BLOCKED: &[&str] = &[
"sys",
"os",
"posix",
"nt",
];
#[rustfmt::skip]
pub(super) const SPECIAL: &[(&[&str], &str)] = &[
(&["torch.nested._internal.nested_tensor.NestedTensor","torch.nested._internal.nested_tensor._rebuild_njt","torch._dynamo.decorators._DimRange"], "``torch.nested`` and ``torch._dynamo`` must be imported to load nested jagged tensors (NJTs)"),
(&["torch.distributed.device_mesh.DeviceMesh","torch.distributed.tensor._dtensor_spec.DTensorSpec","torch.distributed.tensor._dtensor_spec.TensorMeta","torch.distributed.tensor.DTensor","torch.distributed.tensor.placement_types.Partial","torch.distributed.tensor.placement_types.Replicate","torch.distributed.tensor.placement_types.Shard"], "``torch.distributed.tensor`` must be imported to load DTensors"),
];
