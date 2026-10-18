//! Frozen AST/compiler cache-key construction; no executable-cache policy yet.
use crate::triton_frontend::{Count, Element, Fp8Body, Fp8Specialization};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[path = "python_printable.rs"]
mod printable;
/// CPython's str repr over Unicode scalars, including quote choice and its
/// locked Unicode printable table. OS surrogateescape decoding is separate.
pub fn python_str_repr(value: &str) -> String {
    python_codepoint_str_repr(&value.chars().map(u32::from).collect::<Vec<_>>())
}
pub(crate) fn python_codepoint_str_repr(value: &[u32]) -> String {
    use std::fmt::Write;
    let quote = if value.contains(&u32::from('\'')) && !value.contains(&u32::from('"')) {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for &cp in value {
        let Some(c) = char::from_u32(cp) else {
            write!(out, "\\u{cp:04x}").unwrap();
            continue;
        };
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => {
                let cp = c as u32;
                let index = printable::NOT_PRINTABLE.partition_point(|&(start, _)| start <= cp);
                let hidden = index > 0 && cp <= printable::NOT_PRINTABLE[index - 1].1;
                if !hidden {
                    out.push(c);
                } else if cp <= 0xff {
                    write!(out, "\\x{cp:02x}").unwrap();
                } else if cp <= 0xffff {
                    write!(out, "\\u{cp:04x}").unwrap();
                } else {
                    write!(out, "\\U{cp:08x}").unwrap();
                }
            }
        }
    }
    out.push(quote);
    out
}
fn signature(t: Element) -> &'static str {
    match t {
        Element::F64 => "fp64",
        Element::F32 => "fp32",
        Element::F16 => "fp16",
        Element::BF16 => "bf16",
        Element::E4M3 => "fp8e4nv",
        Element::E5M2 => "fp8e5",
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn digest(value: &str) -> String {
    hex(&Sha256::digest(value.as_bytes()))
}
pub(crate) fn source_key_digest(key: &str) -> String {
    digest(key)
}
/// Split the frozen Python signature/attribute representation at outer commas.
fn python_items(text: &str) -> Vec<String> {
    let (mut depth, mut quote, mut escaped, mut start) = (0i32, None, false, 0usize);
    let mut values = vec![];
    for (i, c) in text.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => quote = Some(c),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                let value = text[start..i].trim();
                if !value.is_empty() {
                    values.push(value.into());
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let value = text[start..].trim();
    if !value.is_empty() {
        values.push(value.into());
    }
    values
}
fn source_parts(key: &str) -> (&str, &str, Vec<String>, &str) {
    let (function, rest) = key.split_once("-{").unwrap();
    let (attrs, rest) = rest.split_once("}-[").unwrap();
    let (signature, constants) = rest.split_once("]-").unwrap();
    (function, attrs, python_items(signature), constants)
}
fn attribute_path(entry: &str) -> Vec<usize> {
    entry
        .split_once(')')
        .unwrap()
        .0
        .trim_start_matches('(')
        .split(',')
        .filter_map(|n| {
            let n = n.trim();
            (!n.is_empty()).then(|| n.parse().unwrap())
        })
        .collect()
}
fn constexprs(signature: &str) -> usize {
    if signature == "'constexpr'" {
        1
    } else if signature.starts_with('(') {
        python_items(&signature[1..signature.len() - 1])
            .iter()
            .map(|s| constexprs(s))
            .sum()
    } else {
        0
    }
}
/// Replace a parameter of a frozen flat-prefix kernel signature. Constants in
/// the existing source key are integer/dtype/bool/None values; preserve signs.
pub(crate) fn source_argument_key(
    key: &str,
    names: &[&str],
    index: usize,
    argument: &crate::triton_arguments::ArgumentType,
) -> String {
    let (function, attributes, mut signatures, constants) = source_parts(key);
    let mut sorted = names.to_vec();
    sorted.sort();
    let sorted_index = sorted.binary_search(&names[index]).unwrap();
    let before = names[..index]
        .iter()
        .map(|name| constexprs(&signatures[sorted.binary_search(name).unwrap()]))
        .sum::<usize>();
    let previous = constexprs(&signatures[sorted_index]);
    let mut constants_values = vec![];
    let mut negative = false;
    for part in constants.split('-') {
        if part.is_empty() {
            negative = true;
        } else {
            constants_values.push(format!("{}{part}", if negative { "-" } else { "" }));
            negative = false;
        }
    }
    let mut constant_values = constants_values[..before].to_vec();
    constant_values.extend(argument.constants());
    constant_values.extend_from_slice(&constants_values[before + previous..]);
    signatures[sorted_index] = argument.signature_repr();
    let mut attrs = python_items(attributes)
        .into_iter()
        .filter(|entry| attribute_path(entry)[0] != index)
        .collect::<Vec<_>>();
    attrs.extend(argument.attributes(&[index]));
    attrs.sort_by_key(|entry| attribute_path(entry));
    format!(
        "{function}-{{{}}}-[{}]-{}",
        attrs.join(", "),
        signatures.join(", "),
        constant_values.join("-")
    )
}
/// Replace source constexpr values together, before joining their str values.
/// The input key is generated from the frozen numeric/bool/None prefix; inserted
/// string literals may contain separators and must not be split again.
pub(crate) fn source_constexpr_key(
    key: &str,
    names: &[&str],
    replacements: &[(usize, String)],
) -> String {
    let (function, attributes, signatures, constants) = source_parts(key);
    let mut sorted = names.to_vec();
    sorted.sort();
    let mut values = vec![];
    let mut negative = false;
    for part in constants.split('-') {
        if part.is_empty() {
            negative = true;
        } else {
            values.push(format!("{}{part}", if negative { "-" } else { "" }));
            negative = false;
        }
    }
    for (index, value) in replacements {
        let before = names[..*index]
            .iter()
            .map(|name| constexprs(&signatures[sorted.binary_search(name).unwrap()]))
            .sum::<usize>();
        assert_eq!(
            constexprs(&signatures[sorted.binary_search(&names[*index]).unwrap()]),
            1
        );
        values[before] = value.clone();
    }
    format!(
        "{function}-{{{attributes}}}-[{}]-{}",
        signatures.join(", "),
        values.join("-")
    )
}
/// JITFunction._pack_args replaces root constexpr attributes with an empty
/// string, but nested tuple constexpr values retain their non-string value and
/// are absent from the attribute map. Non-constexpr tuples recurse normally.
pub fn jit_source_digest(key: &str, argument_names: &[&str]) -> String {
    let (function, attributes, signatures, constants) = source_parts(key);
    let mut sorted = argument_names.to_vec();
    sorted.sort();
    assert_eq!(signatures.len(), sorted.len());
    let explicit = python_items(attributes);
    let mut attrs = vec![];
    fn walk(signature: &str, path: &[usize], explicit: &[String], attrs: &mut Vec<String>) {
        if signature.starts_with('(') {
            for (i, value) in python_items(&signature[1..signature.len() - 1])
                .iter()
                .enumerate()
            {
                let mut child = path.to_vec();
                child.push(i);
                walk(value, &child, explicit, attrs);
            }
            return;
        }
        if let Some(value) = explicit.iter().find(|entry| attribute_path(entry) == path) {
            attrs.push(value.clone());
            return;
        }
        let ty = signature.trim_matches('\'');
        if ty.starts_with('*')
            || matches!(ty, "i32" | "i64" | "u32" | "u64")
            || (ty == "constexpr" && path.len() == 1)
        {
            attrs.push(format!("{}: []", crate::triton_arguments::path_repr(path)));
        }
    }
    for (i, name) in argument_names.iter().enumerate() {
        walk(
            &signatures[sorted.binary_search(name).unwrap()],
            &[i],
            &explicit,
            &mut attrs,
        );
    }
    digest(&format!(
        "{function}-{{{}}}-[{}]-{constants}",
        attrs.join(", "),
        signatures.join(", ")
    ))
}
/// Exact original ASTSource.hash for the supported typed frontend component.
pub(crate) fn ast_key(s: &Fp8Specialization<'_>, function_cache_key: &str) -> String {
    let quant = matches!(s.body, Fp8Body::Quantize);
    let mut attrs = vec![];
    for (i, aligned) in s.pointer_divisibility_16.into_iter().enumerate() {
        if aligned {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    let count = match s.count {
        Count::One => "constexpr",
        Count::I32 { .. } => "i32",
        Count::I64 { .. } => "i64",
    };
    if matches!(
        s.count,
        Count::I32 {
            divisible_by_16: true
        } | Count::I64 {
            divisible_by_16: true
        }
    ) {
        let i = if quant { 4 } else { 3 };
        attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
    }
    let attrs = format!("{{{}}}", attrs.join(", "));
    // ASTSource sorts by Python argument name before extracting values.
    let mut sig = vec!["constexpr".to_string()];
    if quant {
        sig.push("fp32".into());
    }
    sig.extend([
        count.into(),
        format!("*{}", signature(s.output)),
        format!("*{}", signature(s.scale)),
        format!("*{}", signature(s.input)),
    ]);
    let sig = format!(
        "[{}]",
        sig.iter()
            .map(|s| python_str_repr(s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let constants = if matches!(s.count, Count::One) {
        format!("1-{}", s.block_size)
    } else {
        s.block_size.to_string()
    };
    format!("{function_cache_key}-{attrs}-{sig}-{constants}")
}
/// Original rowwise ASTSource.hash, including both constexpr arguments.
pub(crate) fn rowwise_ast_key(
    s: &crate::triton_frontend::rowwise::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    let mut attrs = vec![];
    for (i, aligned) in s.pointer_divisibility_16.into_iter().enumerate() {
        if aligned {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    if matches!(
        s.count,
        Count::I32 {
            divisible_by_16: true
        } | Count::I64 {
            divisible_by_16: true
        }
    ) {
        attrs.push("(3,): [['tt.divisibility', 16]]".into());
    }
    let count = match s.count {
        Count::One => "constexpr",
        Count::I32 { .. } => "i32",
        Count::I64 { .. } => "i64",
    };
    let sig = [
        "constexpr".to_string(),
        "constexpr".into(),
        count.into(),
        "*fp32".into(),
        format!("*{}", signature(s.input)),
        "*i8".into(),
    ];
    let sig = format!(
        "[{}]",
        sig.iter()
            .map(|x| python_str_repr(x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let constants = format!(
        "{}{}-{}",
        if matches!(s.count, Count::One) {
            "1-"
        } else {
            ""
        },
        s.block_size,
        s.input_dtype_code
    );
    format!(
        "{function_cache_key}-{{{}}}-{sig}-{constants}",
        attrs.join(", ")
    )
}
/// Original AdaLN signature ordering and shape/dtype/boolean constants.
pub(crate) fn adaln_ast_key(
    s: &crate::triton_frontend::adaln::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    adaln_ast_key_scalar(
        s,
        function_cache_key,
        crate::triton_frontend::scalar::ScalarType::F32,
    )
}
pub(crate) fn adaln_ast_key_scalar(
    s: &crate::triton_frontend::adaln::Specialization<'_>,
    function_cache_key: &str,
    epsilon: crate::triton_frontend::scalar::ScalarType,
) -> String {
    let mut attrs = vec![];
    for (i, aligned) in s.pointer_divisibility_16.into_iter().enumerate() {
        if aligned {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    for (i, count) in s.scalars.into_iter().enumerate() {
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!("({},): [['tt.divisibility', 16]]", i + 4));
        }
    }
    let scalar: [&str; 7] = s.scalars.map(|count| match count {
        Count::One => "constexpr",
        Count::I32 { .. } => "i32",
        Count::I64 { .. } => "i64",
    });
    if epsilon.divisible() {
        attrs.push("(11,): [['tt.divisibility', 16]]".into());
    }
    let sig = vec![
        "constexpr".into(),
        scalar[0].into(),
        "constexpr".into(),
        epsilon.signature().into(),
        scalar[1].into(),
        format!("*{}", signature(s.scale)),
        scalar[2].into(),
        format!("*{}", signature(s.shift)),
        scalar[5].into(),
        scalar[4].into(),
        scalar[3].into(),
        scalar[6].into(),
        "constexpr".into(),
        format!("*{}", signature(s.input)),
        format!("*{}", signature(s.input)),
    ];
    let sig = format!(
        "[{}]",
        sig.iter()
            .map(|s| python_str_repr(s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut constants = s
        .scalars
        .into_iter()
        .filter(|c| matches!(c, Count::One))
        .map(|_| "1".to_string())
        .collect::<Vec<_>>();
    if matches!(epsilon, crate::triton_frontend::scalar::ScalarType::One) {
        constants.push("1".into());
    }
    constants.extend([
        s.block_size.to_string(),
        signature(s.input).into(),
        if s.subtract_mean { "True" } else { "False" }.into(),
    ]);
    format!(
        "{function_cache_key}-{{{}}}-{sig}-{}",
        attrs.join(", "),
        constants.join("-")
    )
}
/// Original flexible RoPE ASTSource hash, including optional K pointer constants.
pub(crate) fn rope_ast_key(
    s: &crate::triton_frontend::rope::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    let mut attrs = vec![];
    let mut sig = BTreeMap::new();
    let mut constants = vec![];
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
            sig.insert(name, format!("*{}", signature(kind)));
            if s.pointer_divisibility_16[i] {
                attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
            }
        } else {
            sig.insert(name, "constexpr".into());
            constants.push("None".to_string());
        }
    }
    for (i, (name, count)) in crate::triton_frontend::rope::SCALAR_NAMES
        .iter()
        .zip(s.scalars)
        .enumerate()
    {
        let kind = match count {
            Count::One => {
                constants.push("1".into());
                "constexpr"
            }
            Count::I32 { .. } => "i32",
            Count::I64 { .. } => "i64",
        };
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!("({},): [['tt.divisibility', 16]]", i + 5));
        }
        sig.insert(name, kind.into());
    }
    for name in ["compute_dtype", "block_size", "split_half"] {
        sig.insert(name, "constexpr".into());
    }
    constants.extend([
        signature(s.frequencies).into(),
        s.block_size.to_string(),
        if s.split_half { "True" } else { "False" }.into(),
    ]);
    let sig = sig
        .values()
        .map(|v| python_str_repr(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{function_cache_key}-{{{}}}-[{sig}]-{}",
        attrs.join(", "),
        constants.join("-")
    )
}
/// Original RMS-RoPE ASTSource hash and normalization materialization dtype.
pub(crate) fn rms_rope_ast_key(
    s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    rms_rope_ast_key_scalar(
        s,
        function_cache_key,
        crate::triton_frontend::scalar::ScalarType::F32,
    )
}
pub(crate) fn rms_rope_ast_key_scalar(
    s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    function_cache_key: &str,
    epsilon: crate::triton_frontend::scalar::ScalarType,
) -> String {
    let mut attrs = vec![];
    let mut sig = BTreeMap::new();
    let mut constants = vec![];
    for (i, (name, kind)) in [
        ("x_ptr", s.input),
        ("freqs_ptr", s.frequencies),
        ("scale_ptr", s.scale),
        ("out_ptr", s.input),
    ]
    .into_iter()
    .enumerate()
    {
        sig.insert(name, format!("*{}", signature(kind)));
        if s.pointer_divisibility_16[i] {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    for (i, (name, count)) in crate::triton_frontend::rope::rms::SCALAR_NAMES
        .iter()
        .zip(s.scalars)
        .enumerate()
    {
        let kind = match count {
            Count::One => {
                constants.push("1".to_string());
                "constexpr"
            }
            Count::I32 { .. } => "i32",
            Count::I64 { .. } => "i64",
        };
        sig.insert(name, kind.into());
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!("({},): [['tt.divisibility', 16]]", i + 4));
        }
    }
    sig.insert("epsilon", epsilon.signature().into());
    if matches!(epsilon, crate::triton_frontend::scalar::ScalarType::One) {
        constants.push("1".into());
    }
    if epsilon.divisible() {
        attrs.push("(20,): [['tt.divisibility', 16]]".into());
    }
    for name in ["compute_dtype", "norm_dtype", "block_size", "split_half"] {
        sig.insert(name, "constexpr".into());
    }
    constants.extend([
        signature(s.frequencies).into(),
        signature(s.input).into(),
        s.block_size.to_string(),
        if s.split_half { "True" } else { "False" }.into(),
    ]);
    let sig = sig
        .values()
        .map(|v| python_str_repr(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{function_cache_key}-{{{}}}-[{sig}]-{}",
        attrs.join(", "),
        constants.join("-")
    )
}
/// Original MXFP8 pointer/scalar signature and both launch constants.
pub(crate) fn mxfp8_ast_key(
    s: &crate::triton_frontend::mxfp8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    let mut attrs = vec![];
    let mut sig = BTreeMap::new();
    let mut constants = vec![];
    for (i, (name, kind)) in [
        ("x_ptr", format!("*{}", signature(s.input))),
        ("output_ptr", "*fp8e4nv".into()),
        ("swizzled_scales_ptr", "*u8".into()),
    ]
    .into_iter()
    .enumerate()
    {
        sig.insert(name, kind);
        if s.pointer_divisibility_16[i] {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    for (i, (name, count)) in crate::triton_frontend::mxfp8::SCALAR_NAMES
        .iter()
        .zip(s.scalars)
        .enumerate()
    {
        let kind = match count {
            Count::One => {
                constants.push("1".to_string());
                "constexpr"
            }
            Count::I32 { .. } => "i32",
            Count::I64 { .. } => "i64",
        };
        sig.insert(name, kind.into());
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!("({},): [['tt.divisibility', 16]]", i + 3));
        }
    }
    for name in ["block_size", "blocks_per_program"] {
        sig.insert(name, "constexpr".into());
    }
    constants.extend([s.block_size.to_string(), s.blocks_per_program.to_string()]);
    let sig = sig
        .values()
        .map(|v| python_str_repr(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{function_cache_key}-{{{}}}-[{sig}]-{}",
        attrs.join(", "),
        constants.join("-")
    )
}
/// Original ASTSource hashes for both NVFP4 source kernels.
pub(crate) fn nvfp4_ast_key(
    s: &crate::triton_frontend::nvfp4::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    block_quant_ast_key(
        function_cache_key,
        &[
            ("x_ptr", format!("*{}", signature(s.input))),
            ("packed_output_ptr", "*u8".into()),
            ("swizzled_scales_ptr", "*fp8e4nv".into()),
            (
                "per_tensor_scale_ptr",
                format!("*{}", signature(s.global_scale)),
            ),
        ],
        &s.pointer_divisibility_16,
        &crate::triton_frontend::nvfp4::SCALAR_NAMES,
        &s.scalars,
        &[
            ("block_size", s.block_size.to_string()),
            ("blocks_per_program", s.blocks_per_program.to_string()),
            ("hi_first", if s.hi_first { "True" } else { "False" }.into()),
        ],
    )
}
pub(crate) fn dequant_nvfp4_ast_key(
    s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    block_quant_ast_key(
        function_cache_key,
        &[
            ("packed_ptr", "*u8".into()),
            ("scale_ptr", format!("*{}", signature(s.block_scale))),
            (
                "global_scale_ptr",
                format!("*{}", signature(s.global_scale)),
            ),
            ("output_ptr", format!("*{}", signature(s.output))),
        ],
        &s.pointer_divisibility_16,
        &crate::triton_frontend::nvfp4::dequantize::SCALAR_NAMES,
        &s.scalars,
        &[
            ("block_size", s.block_size.to_string()),
            ("tile_size", s.tile_size.to_string()),
            ("hi_first", if s.hi_first { "True" } else { "False" }.into()),
        ],
    )
}
fn block_quant_ast_key(
    function_cache_key: &str,
    pointers: &[(&str, String)],
    align: &[bool],
    names: &[&str],
    counts: &[Count],
    parameters: &[(&str, String)],
) -> String {
    let mut attrs = vec![];
    let mut sig = BTreeMap::new();
    let mut constants = vec![];
    for (i, (name, kind)) in pointers.iter().enumerate() {
        sig.insert(*name, kind.clone());
        if align[i] {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    for (i, (name, count)) in names.iter().zip(counts).enumerate() {
        let kind = match count {
            Count::One => {
                constants.push("1".to_string());
                "constexpr"
            }
            Count::I32 { .. } => "i32",
            Count::I64 { .. } => "i64",
        };
        sig.insert(*name, kind.into());
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!(
                "({},): [['tt.divisibility', 16]]",
                i + pointers.len()
            ));
        }
    }
    for (name, value) in parameters {
        sig.insert(*name, "constexpr".into());
        constants.push(value.clone());
    }
    let sig = sig
        .values()
        .map(|v| python_str_repr(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{function_cache_key}-{{{}}}-[{sig}]-{}",
        attrs.join(", "),
        constants.join("-")
    )
}

/// Original ASTSource hash, independent of the selected compiler options hash.
pub(crate) fn int8_ast_key(
    s: &crate::triton_frontend::int8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    block_quant_ast_key(
        function_cache_key,
        &[
            ("a_ptr", "*i8".into()),
            ("b_ptr", "*i8".into()),
            ("c_ptr", format!("*{}", signature(s.output))),
            ("a_scale_ptr", "*fp32".into()),
            ("b_scale_ptr", format!("*{}", signature(s.weight_scale))),
            ("bias_ptr", format!("*{}", signature(s.bias))),
        ],
        &s.pointer_divisibility_16,
        &crate::triton_frontend::int8::SCALAR_NAMES,
        &s.scalars,
        &[
            ("block_m", s.block_m.to_string()),
            ("block_n", s.block_n.to_string()),
            ("block_k", s.block_k.to_string()),
            ("group_size_m", s.group_size_m.to_string()),
            ("has_bias", if s.has_bias { "True" } else { "False" }.into()),
        ],
    )
}

pub struct CompilerHashInputs<'a> {
    pub triton_version_key: &'a str,
    pub backend_hash: &'a str,
    pub options_hash: &'a str,
}
pub fn environment_repr(environment: &BTreeMap<String, String>) -> String {
    format!(
        "[{}]",
        environment
            .iter()
            .map(|(k, v)| format!("({}, {})", python_str_repr(k), python_str_repr(v)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
pub fn compiler_digest(
    inputs: &CompilerHashInputs<'_>,
    ast_digest: &str,
    environment: &BTreeMap<String, String>,
) -> String {
    digest(&format!(
        "{}-{ast_digest}-{}-{}-{}",
        inputs.triton_version_key,
        inputs.backend_hash,
        inputs.options_hash,
        environment_repr(environment)
    ))
}
/// Original RFC 4648 base32 encoding of the compiler's hexadecimal digest,
/// without '=' padding, used by all three original local cache modes.
pub fn cache_directory(hex_digest: &str) -> Result<String, crate::native::Error> {
    let mut bytes = Vec::new();
    if !hex_digest.len().is_multiple_of(2) || !hex_digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(crate::native::Error::Native(
            "Expected a compiler hex digest".into(),
        ));
    }
    for i in (0..hex_digest.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&hex_digest[i..i + 2], 16).unwrap());
    }
    const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let (mut bits, mut count, mut out) = (0u32, 0u32, String::new());
    for b in bytes {
        bits = (bits << 8) | u32::from(b);
        count += 8;
        while count >= 5 {
            count -= 5;
            out.push(ALPHABET[((bits >> count) & 31) as usize] as char);
        }
    }
    if count != 0 {
        out.push(ALPHABET[((bits << (5 - count)) & 31) as usize] as char);
    }
    Ok(out)
}

/// Original grouped unpack AST key, including both bit widths and codebook branch.
pub(crate) fn w4a8_ast_key(
    s: &crate::triton_frontend::w4a8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    block_quant_ast_key(
        function_cache_key,
        &[
            ("qdata_ptr", "*i8".into()),
            ("srel_ptr", "*fp32".into()),
            ("cb_ptr", "*fp32".into()),
            ("out_ptr", "*i8".into()),
        ],
        &s.pointer_divisibility_16,
        &crate::triton_frontend::w4a8::SCALAR_NAMES,
        &s.scalars,
        &[
            ("has_cb", if s.has_cb { "True" } else { "False" }.into()),
            ("bits", s.bits.to_string()),
            ("block_n", s.block_n.to_string()),
            ("block_kh", s.block_kh.to_string()),
        ],
    )
}

/// Frozen NA3D AST signature, including scalar-scale specialization.
pub(crate) fn na_ast_key(
    s: &crate::triton_frontend::na::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    use crate::triton_frontend::na::Scale;
    let mut attrs = vec![];
    let mut sig = BTreeMap::new();
    let mut constants = vec![];
    for (i, (name, element)) in [
        ("q_ptr", s.input),
        ("k_ptr", s.key),
        ("v_ptr", s.value),
        ("out_ptr", s.input),
    ]
    .into_iter()
    .enumerate()
    {
        sig.insert(name, format!("*{}", signature(element)));
        if s.pointer_divisibility_16[i] {
            attrs.push(format!("({i},): [['tt.divisibility', 16]]"));
        }
    }
    for (i, (name, count)) in crate::triton_frontend::na::SCALAR_NAMES
        .iter()
        .zip(s.scalars)
        .enumerate()
    {
        let kind = match count {
            Count::One => {
                constants.push("1".into());
                "constexpr"
            }
            Count::I32 { .. } => "i32",
            Count::I64 { .. } => "i64",
        };
        sig.insert(*name, kind.into());
        if matches!(
            count,
            Count::I32 {
                divisible_by_16: true
            } | Count::I64 {
                divisible_by_16: true
            }
        ) {
            attrs.push(format!("({},): [['tt.divisibility', 16]]", i + 4));
        }
    }
    let scale = match s.scale {
        Scale::F32 => "fp32",
        Scale::F64 => "fp64",
        Scale::I32 { .. } => "i32",
        Scale::I64 { .. } => "i64",
        Scale::U64 { .. } => "u64",
        Scale::One => {
            constants.push("1".into());
            "constexpr"
        }
        Scale::Bool => "u1",
    };
    sig.insert("scale", scale.into());
    if matches!(
        s.scale,
        Scale::I32 {
            divisible_by_16: true
        } | Scale::I64 {
            divisible_by_16: true
        } | Scale::U64 {
            divisible_by_16: true
        }
    ) {
        attrs.push("(13,): [['tt.divisibility', 16]]".into());
    }
    for name in [
        "kt", "kh", "kw", "causal_t", "causal_h", "causal_w", "hd", "hd_pad", "block_q", "block_k",
        "is_fp32",
    ] {
        sig.insert(name, "constexpr".into());
    }
    constants.extend([
        s.kt.to_string(),
        s.kh.to_string(),
        s.kw.to_string(),
        if s.causal_t { "True" } else { "False" }.into(),
        if s.causal_h { "True" } else { "False" }.into(),
        if s.causal_w { "True" } else { "False" }.into(),
        s.hd.to_string(),
        s.hd_pad.to_string(),
        s.block_q.to_string(),
        s.block_k.to_string(),
        if s.input == Element::F32 {
            "True"
        } else {
            "False"
        }
        .into(),
    ]);
    let sig = sig
        .values()
        .map(|v| python_str_repr(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{function_cache_key}-{{{}}}-[{sig}]-{}",
        attrs.join(", "),
        constants.join("-")
    )
}

pub fn ast_digest(s: &Fp8Specialization<'_>, function_cache_key: &str) -> String {
    digest(&ast_key(s, function_cache_key))
}
pub fn rowwise_ast_digest(
    s: &crate::triton_frontend::rowwise::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&rowwise_ast_key(s, function_cache_key))
}
pub fn adaln_ast_digest(
    s: &crate::triton_frontend::adaln::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&adaln_ast_key(s, function_cache_key))
}
pub fn rope_ast_digest(
    s: &crate::triton_frontend::rope::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&rope_ast_key(s, function_cache_key))
}
pub fn rms_rope_ast_digest(
    s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&rms_rope_ast_key(s, function_cache_key))
}
pub fn mxfp8_ast_digest(
    s: &crate::triton_frontend::mxfp8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&mxfp8_ast_key(s, function_cache_key))
}
pub fn nvfp4_ast_digest(
    s: &crate::triton_frontend::nvfp4::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&nvfp4_ast_key(s, function_cache_key))
}
pub fn dequant_nvfp4_ast_digest(
    s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&dequant_nvfp4_ast_key(s, function_cache_key))
}
pub fn int8_ast_digest(
    s: &crate::triton_frontend::int8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&int8_ast_key(s, function_cache_key))
}
pub fn w4a8_ast_digest(
    s: &crate::triton_frontend::w4a8::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&w4a8_ast_key(s, function_cache_key))
}
pub fn na_ast_digest(
    s: &crate::triton_frontend::na::Specialization<'_>,
    function_cache_key: &str,
) -> String {
    digest(&na_ast_key(s, function_cache_key))
}

pub fn adaln_ast_digest_scalar(
    s: &crate::triton_frontend::adaln::Specialization<'_>,
    function_cache_key: &str,
    epsilon: crate::triton_frontend::scalar::ScalarType,
) -> String {
    digest(&adaln_ast_key_scalar(s, function_cache_key, epsilon))
}
pub fn rms_rope_ast_digest_scalar(
    s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    function_cache_key: &str,
    epsilon: crate::triton_frontend::scalar::ScalarType,
) -> String {
    digest(&rms_rope_ast_key_scalar(s, function_cache_key, epsilon))
}

/// Explicit ASTSource identity for the original dynamic NVFP4 constexpr.
pub fn nvfp4_literal_ast_digest(
    s: &crate::triton_frontend::nvfp4::Specialization<'_>,
    function_cache_key: &str,
    argument_names: &[&str],
    literal: &str,
) -> String {
    let key = nvfp4_ast_key(s, function_cache_key);
    let index = argument_names
        .iter()
        .position(|n| *n == "hi_first")
        .unwrap();
    digest(&source_constexpr_key(
        &key,
        argument_names,
        &[(index, literal.into())],
    ))
}
/// Explicit ASTSource identity for the inverse dynamic NVFP4 constexpr.
pub fn dequant_nvfp4_literal_ast_digest(
    s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
    function_cache_key: &str,
    argument_names: &[&str],
    literal: &str,
) -> String {
    let key = dequant_nvfp4_ast_key(s, function_cache_key);
    let index = argument_names
        .iter()
        .position(|n| *n == "hi_first")
        .unwrap();
    digest(&source_constexpr_key(
        &key,
        argument_names,
        &[(index, literal.into())],
    ))
}
