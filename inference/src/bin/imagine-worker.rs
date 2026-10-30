//! Private length-bounded pipe protocol; prompts/pixels never become files or argv.
use imagine_inference::{
    checkpoint::{Aimdo, bind_comfy_layouts},
    execution::Runtime,
    native::{Error, Kitchen},
    server::{self, Event, Request},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};
const MAX_HEADER: usize = 1024 * 1024;
const MAX_PAYLOAD: usize = 256 * 1024 * 1024;
fn path(key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(std::env::var_os(key)
        .ok_or("Missing native runtime dependency")?
        .into())
}
fn read(reader: &mut impl Read, max: usize) -> io::Result<Vec<u8>> {
    let mut len = [0; 4];
    reader.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    if n > max {
        return Err(io::Error::other("Frame exceeds bound"));
    }
    let mut data = vec![0; n];
    reader.read_exact(&mut data)?;
    Ok(data)
}
fn write(out: &mut impl Write, event: Value, data: &[u8]) -> io::Result<()> {
    let json = serde_json::to_vec(&event)?;
    if json.len() > MAX_HEADER || data.len() > MAX_PAYLOAD {
        return Err(io::Error::other("Frame exceeds bound"));
    }
    out.write_all(&(json.len() as u32).to_be_bytes())?;
    out.write_all(&json)?;
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(data)?;
    out.flush()
}
/// The `{kind}_lengths`/`{kind}_shapes` RGBA8 images from `offset` in the payload,
/// as `[1, h, w, 4]` in [0, 1]; `reference` lists are required, others optional.
fn rgba_images(
    header: &Value,
    kind: &str,
    payload: &[u8],
    offset: &mut usize,
    max: usize,
) -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    let (lengths, shapes) = match (
        header[format!("{kind}_lengths")].as_array(),
        header[format!("{kind}_shapes")].as_array(),
    ) {
        (Some(lengths), Some(shapes)) => (lengths, shapes),
        (None, None) if kind != "reference" => return Ok(Vec::new()),
        _ => return Err(format!("Missing {kind} images").into()),
    };
    if shapes.len() != lengths.len() {
        return Err(format!("{kind} shape count").into());
    }
    if lengths.len() > max {
        return Err(format!("Too many {kind} images").into());
    }
    let mut images = vec![];
    for (i, n) in lengths.iter().enumerate() {
        let n = n.as_u64().ok_or("Invalid image length")? as usize;
        let end = offset
            .checked_add(n)
            .filter(|n| *n <= payload.len())
            .ok_or("Invalid image boundary")?;
        let side = |j: usize| {
            shapes[i][j]
                .as_i64()
                .filter(|n| *n > 0 && *n <= 8192)
                .ok_or("Invalid image shape")
        };
        let (h, w) = (side(0)?, side(1)?);
        if n != h as usize * w as usize * 4 {
            return Err("Invalid RGBA image size".into());
        }
        let values = payload[*offset..end]
            .iter()
            .map(|v| *v as f32 / 255.)
            .collect::<Vec<_>>();
        images.push(Tensor::f_from_slice(&values)?.f_view([1, h, w, 4])?);
        *offset = end;
    }
    Ok(images)
}
fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let native = unsafe { Kitchen::load(&path("IMAGINE_KITCHEN_LIBRARY")?) }?;
    let cache = path("IMAGINE_WORKER_CACHE")?;
    std::fs::create_dir_all(&cache)?;
    let compiler = Arc::new(unsafe {
        Compiler::open(
            &path("IMAGINE_TRITON_COMPILER_PROFILE")?,
            &cache.join("compiler"),
            &cache.join("work"),
            native.clone(),
        )
    }?);
    let rt = Runtime::frozen(native, compiler)?;
    rt.backends.policy.disable("triton");
    let mut sdpa = rt.native.sdpa_state()?;
    sdpa[..3].copy_from_slice(&[1, 1, 1]);
    rt.native.set_sdpa_state(sdpa)?;
    let mut math = rt.native.comfy_math_state()?;
    math[0] = 1;
    math[1] = 0;
    rt.native.set_comfy_math_state(math)?;
    bind_comfy_layouts(&rt);
    let aimdo = unsafe { Aimdo::load(&path("IMAGINE_AIMDO_LIBRARY")?) }?;
    rt.native.cuda_memory(0)?;
    // The previous job's dynamic models, until a job needs others or the
    // server stops this worker to release the GPU.
    let mut kept = server::Kept::default();
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    write(
        &mut output,
        json!({"type":"ready","protocol":1,"attention":"checkpoint_preferences"}),
        &[],
    )?;
    loop {
        let header = match read(&mut input, MAX_HEADER) {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let payload = read(&mut input, MAX_PAYLOAD)?;
        let value: Value = serde_json::from_slice(&header)?;
        if value["protocol"] != 1 || !matches!(value["type"].as_str(), Some("generate" | "maps")) {
            return Err("Invalid frame".into());
        }
        if value["type"] == "maps" {
            let id = value["id"]
                .as_str()
                .ok_or("Missing request identity")?
                .to_owned();
            // The photo arrives as RGBA8; the maps return as RGB8, depth first.
            let result = (|| -> Result<(Value, Vec<u8>), Box<dyn std::error::Error>> {
                let request: server::maps::MapsRequest = serde_json::from_value(value.clone())?;
                let [h, w] = request.shape;
                if h <= 0 || w <= 0 || h > 8192 || w > 8192 || payload.len() as i64 != h * w * 4 {
                    return Err("Invalid RGBA photo size".into());
                }
                let rgb: Vec<u8> = payload
                    .chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                let maps = server::maps::control_maps(&request, &rgb)?;
                rt.native.cuda_soft_empty_cache(0)?;
                let mut bytes = maps.depth;
                bytes.extend(maps.regions);
                Ok((json!({"type":"result","id":id}), bytes))
            })();
            match result {
                Ok((header, bytes)) => write(&mut output, header, &bytes)?,
                Err(_) => {
                    write(
                        &mut output,
                        json!({"type":"error","id":id,"code":"rust_control_maps_failed"}),
                        &[],
                    )?;
                    return Ok(());
                }
            }
            continue;
        }
        let id = value["id"]
            .as_str()
            .ok_or("Missing request identity")?
            .to_owned();
        let mut load_failure = None;
        let mut attention_unavailable = false;
        let result = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let request: Request = serde_json::from_value(value.clone())?;
            // An explicit Kitchen INT8 choice fails by name, before any loading.
            if server::attention_method(&rt, request.execution)?.is_none() {
                attention_unavailable = true;
                return Err("Comfy Kitchen INT8 attention is unavailable".into());
            }
            let mut offset = 0usize;
            let images = rgba_images(&value, "reference", &payload, &mut offset, 10)?;
            let roles = value["map_roles"].as_array().cloned().unwrap_or_default();
            let mut maps = server::ControlMaps::default();
            if !roles.is_empty() {
                let pixels = rgba_images(&value, "map", &payload, &mut offset, 2)?;
                if pixels.len() != roles.len() {
                    return Err("Map role count".into());
                }
                for (role, pixels) in roles.iter().zip(pixels) {
                    let slot = match role.as_str() {
                        Some("color") => &mut maps.color,
                        Some("depth") => &mut maps.depth,
                        _ => return Err("Invalid map role".into()),
                    };
                    if slot.replace(pixels).is_some() {
                        return Err("Repeated map role".into());
                    }
                }
            }
            if offset != payload.len() {
                return Err("Trailing image data".into());
            }
            let start = Instant::now();
            let mut observer = |event| {
                let event = match event {
                    Event::Preview { metadata, pixels } => {
                        return write(
                            &mut output,
                            json!({"type":"preview","id":id,"metadata":metadata}),
                            &pixels,
                        )
                        .map_err(|e| Error::Native(e.to_string()));
                    }
                    Event::PreviewWarning => json!({"type":"preview_warning","id":id}),
                    Event::LoadFailure(diagnostic) => {
                        load_failure = Some(diagnostic);
                        return Ok(());
                    }
                    Event::Phase(phase) => json!({"type":"phase","id":id,"phase":phase}),
                    Event::Step(i) => {
                        json!({"type":"progress","id":id,"completed":i,"total":request.sampling.steps,"elapsed_seconds":start.elapsed().as_secs_f64()})
                    }
                };
                write(&mut output, event, &[]).map_err(|e| Error::Native(e.to_string()))
            };
            let pixels = server::generate(
                &rt,
                &aimdo,
                &request,
                &images,
                &maps,
                &mut kept,
                &mut observer,
            )?
            .f_get(0)?
            .f_to_device(Device::Cpu)?
            .f_mul_scalar(255.)?
            .f_to_kind(Kind::Uint8)?
            .f_contiguous()?;
            let shape = pixels.size();
            let mut bytes = vec![0u8; pixels.numel()];
            let n = bytes.len();
            pixels.f_copy_data_u8(&mut bytes, n)?;
            if !matches!(shape[2], 3 | 4) {
                return Err("Invalid output channels".into());
            }
            Ok(bytes)
        })();
        match result {
            Ok(pixels) => write(&mut output, json!({"type":"result","id":id}), &pixels)?,
            Err(_) => {
                let event = match load_failure {
                    _ if attention_unavailable => {
                        json!({"type":"error","id":id,"code":"attention_unavailable"})
                    }
                    Some(diagnostic) => {
                        json!({"type":"error","id":id,"code":"model_component_load_failed","diagnostic":diagnostic})
                    }
                    None => json!({"type":"error","id":id,"code":"rust_generation_failed"}),
                };
                write(&mut output, event, &[])?;
                return Ok(());
            }
        }
    }
}
fn main() {
    // Never print third-party exceptions, file paths or user inputs to a log.
    if serve().is_err() {
        let _ = write(
            &mut io::stdout().lock(),
            json!({"type":"error","id":null,"code":"rust_startup_failed"}),
            &[],
        );
        std::process::exit(1)
    }
}
