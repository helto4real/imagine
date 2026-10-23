// Appended only to a separate generated copy of native.rs for owned diagnostics.
fn probe_root() -> NResult<PathBuf> {
    let root = PathBuf::from(std::env::var_os("IMAGINE_H3_EXP_STEP").ok_or_else(native_error)?);
    if !root.starts_with("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
        || root.canonicalize().map_err(|_| native_error())? != root
    {
        return Err(native_error());
    }
    Ok(root)
}
pub(crate) fn dump(name: &str, value: &Tensor) -> NResult<()> {
    use std::io::Write;
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(native_error());
    }
    let root = probe_root()?.join("rust");
    let cpu = value
        .f_to_device_(Device::Cpu, Kind::Float, false, false)?
        .f_contiguous()?;
    let mut values = vec![0f32; cpu.numel()];
    let n = values.len();
    cpu.f_copy_data(&mut values, n)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{name}.f32")))
        .map_err(|_| native_error())?;
    file.write_all(&bytes).map_err(|_| native_error())?;
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "shape": value.size(), "original_dtype": format!("{:?}",value.kind()),
            "storage_dtype":"float32", "sha256":format!("{:x}",Sha256::digest(&bytes)),
            "finite": values.iter().all(|v|v.is_finite())
        }))
        .map_err(|_| native_error())?,
    )
    .map_err(|_| native_error())?;
    Ok(())
}
fn probe_observe(name: &str, value: &Tensor) -> NResult<()> {
    if matches!(
        name,
        "h3-packed"
            | "h3-time-basis"
            | "h3-rope"
            | "h3-block-0"
            | "h3.sampling.0.x"
            | "h3.sampling.0.denoised"
    ) {
        dump(name, value)?;
    }
    if name == "h3.sampling.0.denoised" {
        return Err(Error::Native("OWN_EXPECTED_FIRST_MASKED_STEP_STOP".into()));
    }
    Ok(())
}
pub fn probe() -> Result<(), String> {
    let root = probe_root().map_err(|e| format!("{e:?}"))?;
    let mut request: Request = serde_json::from_slice(
        &std::fs::read(root.join("request.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    request.continuation_bytes =
        std::fs::read(root.join("source-tail.f32")).map_err(|e| e.to_string())?;
    let input = request
        .continuation
        .as_ref()
        .ok_or("missing own continuation")?;
    if !request.valid()
        || !input.descriptor.valid_bytes(&request.continuation_bytes)
        || format!("{:x}", Sha256::digest(&request.continuation_bytes)) != input.sha256
    {
        return Err("invalid own continuation".into());
    }
    let config = Config::environment().map_err(|e| format!("{e:?}"))?;
    let mut native = Native { config };
    let mut events = vec![];
    let result = native.generate(&request, &mut |event, bytes| {
        if !bytes.is_empty() {
            return Err(ErrorCode::GenerationFailed);
        };
        events.push(serde_json::to_value(event).map_err(|_| ErrorCode::GenerationFailed)?);
        Ok(())
    });
    let expected = matches!(result, Err(ErrorCode::GenerationFailed))
        && root.join("rust/h3.sampling.0.denoised.json").is_file();
    std::fs::write(root.join("rust-result.json"),serde_json::to_vec_pretty(&serde_json::json!({"expected_first_step_stop":expected,"events":events,"result":format!("{result:?}"),"no_decode_or_export":true})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if expected {
        Ok(())
    } else {
        Err(format!("Unexpected probe result: {result:?}"))
    }
}
