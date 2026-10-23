use imagine_h3_product_cpu::{
    deployment,
    protocol::{self as p, ErrorCode, Event, Request, Stream},
    wire::{self, Generator},
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Own(PathBuf);
impl Own {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "h3-cpu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&p).unwrap();
        std::fs::write(p.join("cpu-dependency.txt"), b"own non-GPU source fixture").unwrap();
        Self(p)
    }
}
impl Drop for Own {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn request(prompt: &str) -> Request {
    Request {
        capture_continuation: false,
        continuation: None,
        continuation_bytes: vec![],
        references: Default::default(),
        sampling: None,
        protocol: p::VERSION,
        kind: p::RequestKind::GenerateVideo,
        id: "cpu-request".into(),
        profile: p::Profile::ShortAvNoReferenceV1,
        prompt: prompt.into(),
        shape: Default::default(),
        image: None,
        image_rgb: Vec::new(),
        seed: 42,
    }
}
fn frame(header: &[u8], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&(header.len() as u32).to_be_bytes());
    v.extend_from_slice(header);
    v.extend_from_slice(&(body.len() as u32).to_be_bytes());
    v.extend_from_slice(body);
    v
}

#[test]
fn reference_rgb_and_pcm_reach_worker_in_manifest_order() {
    let mut r = request("Own Picture 1 and Audio 1 reference");
    r.profile = p::Profile::AvNoReferenceV2;
    r.references.manifest = serde_json::json!({"images":[{"prompt_label":"<Picture 1>"}],"audios":[{"prompt_label":"<Audio 1>"}]});
    r.references.images.push(p::references::Image {
        width: 2,
        height: 1,
        rgb: vec![3, 17, 91, 201, 8, 66],
    });
    let pcm = [0.1f32, -0.2, 0.3, -0.4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    r.references.audios.push(p::references::Audio {
        sample_rate: 44100,
        samples: 2,
        pcm: pcm.clone(),
    });
    let video_pcm = [0.9f32, -0.8]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    r.references.videos.push(p::references::Video {
        width: 2,
        height: 1,
        frames: 5,
        source_fps: 24.,
        tail_frames: None,
        rgb: [19, 21, 14, 99, 180, 70].repeat(5),
        audio: Some(p::references::Audio {
            sample_rate: 44100,
            samples: 1,
            pcm: video_pcm.clone(),
        }),
    });
    let header = serde_json::to_vec(&r).unwrap();
    let mut body = r.references.images[0].rgb.clone();
    body.extend_from_slice(&r.references.videos[0].rgb);
    body.extend_from_slice(&video_pcm);
    body.extend_from_slice(&pcm);
    let parsed = wire::request(&mut std::io::Cursor::new(frame(&header, &body))).unwrap();
    assert_eq!(parsed.references.images[0].rgb, r.references.images[0].rgb);
    assert_eq!(parsed.references.videos[0].rgb, r.references.videos[0].rgb);
    assert_eq!(
        parsed.references.videos[0].audio.as_ref().unwrap().pcm,
        video_pcm
    );
    assert_eq!(parsed.references.audios[0].pcm, pcm);
    assert_eq!(parsed.references.manifest, r.references.manifest);
    assert!(parsed.references.valid_payload());
    body.pop();
    assert!(wire::request(&mut std::io::Cursor::new(frame(&header, &body))).is_err());
}
fn encoded(r: &Request) -> Vec<u8> {
    frame(&serde_json::to_vec(r).unwrap(), &[])
}
fn event(input: &mut impl Read) -> (Event, Vec<u8>) {
    let mut len = [0; 4];
    input.read_exact(&mut len).unwrap();
    let n = u32::from_be_bytes(len) as usize;
    assert!(n <= p::MAX_HEADER);
    let mut header = vec![0; n];
    input.read_exact(&mut header).unwrap();
    input.read_exact(&mut len).unwrap();
    let n = u32::from_be_bytes(len) as usize;
    assert!(n <= p::MAX_PAYLOAD);
    let mut body = vec![0; n];
    input.read_exact(&mut body).unwrap();
    (serde_json::from_slice(&header).unwrap(), body)
}
fn ready(own: &Own) -> p::Deployment {
    let exe = std::env::current_exe().unwrap();
    let path = own.0.join("cpu-dependency.txt");
    let map = BTreeMap::from([(
        path.to_str().unwrap().into(),
        deployment::sha(&path).unwrap(),
    )]);
    deployment::identity(&exe, &map, &[], &[]).unwrap()
}
struct Never;
impl Generator for Never {
    fn generate(&mut self, _: &Request, _: &mut wire::Emit<'_>) -> Result<(), ErrorCode> {
        panic!("not admitted")
    }
}
#[test]
fn invalid_request_rejects_before_factory_or_native_initialization() {
    let own = Own::new();
    let good = request("SYNTHETIC_PRIVATE_CANARY");
    let mut cases = Vec::new();
    let mut value = serde_json::to_value(&good).unwrap();
    value["seed"] = 43.into();
    cases.push(frame(&serde_json::to_vec(&value).unwrap(), &[]));
    for (key, v) in [
        ("protocol", serde_json::json!(p::VERSION + 1)),
        (
            "shape",
            serde_json::json!({"width":384,"height":256,"frames":40,"fps":24}),
        ),
        ("profile", serde_json::json!("unknown")),
        ("reference", serde_json::json!("forbidden")),
        ("prompt", serde_json::json!(" ")),
        ("id", serde_json::json!("invalid id")),
    ] {
        let mut value = serde_json::to_value(&good).unwrap();
        value[key] = v;
        cases.push(frame(&serde_json::to_vec(&value).unwrap(), &[]));
    }
    cases.extend([
        vec![],
        vec![0, 1],
        ((p::MAX_HEADER + 1) as u32).to_be_bytes().to_vec(),
        frame(b"not-json", &[]),
        frame(&serde_json::to_vec(&good).unwrap(), b"not empty"),
    ]);
    let mut long = request(&"a".repeat(p::MAX_PROMPT_BYTES + 1));
    cases.push(encoded(&long));
    long.prompt = "\0".repeat(p::MAX_PROMPT_BYTES + 1);
    cases.push(encoded(&long));
    for input in cases {
        let mut called = false;
        let mut output = Vec::new();
        let result = wire::serve(&mut input.as_slice(), &mut output, ready(&own), |_| {
            called = true;
            Ok(Never)
        });
        assert_eq!(result, Err(ErrorCode::InvalidRequest));
        assert!(!called);
        let mut input = output.as_slice();
        assert!(matches!(event(&mut input).0, Event::Ready { .. }));
        assert!(matches!(
            event(&mut input).0,
            Event::Error {
                id: None,
                code: ErrorCode::InvalidRequest
            }
        ));
        assert!(input.is_empty());
        assert!(
            !output
                .windows(b"SYNTHETIC_PRIVATE_CANARY".len())
                .any(|v| v == b"SYNTHETIC_PRIVATE_CANARY")
        );
    }
}
#[test]
fn bounded_stage_header_accepts_escaped_prompt_without_constructing_native_runtime() {
    let request = request(&"\0".repeat(p::MAX_PROMPT_BYTES));
    let header = serde_json::to_vec(&request).unwrap();
    assert!(header.len() > 64 * 1024 && header.len() <= p::MAX_HEADER);
    let bytes = frame(&header, &[]);
    let parsed = wire::request(&mut bytes.as_slice()).unwrap();
    assert_eq!(parsed.prompt, request.prompt);
    assert!(parsed.valid());
}
#[test]
fn latent_tail_transport_validates_version_geometry_length_hash_and_finiteness_before_factory() {
    use sha2::Digest;
    let own = Own::new();
    let mut r = request("own motion fixture");
    r.profile = p::Profile::AvNoReferenceV2;
    let descriptor = p::continuation::Descriptor {
        schema_version: 1,
        shape: r.shape,
        sampled_frames: r.shape.frames,
        visible_end_frame: r.shape.frames,
        selected_end_frame: r.shape.frames,
    };
    let body = vec![0; descriptor.bytes()];
    r.continuation = Some(p::continuation::Input {
        method: p::continuation::Method::MotionContextExp,
        include_audio: true,
        descriptor,
        sha256: format!("{:x}", sha2::Sha256::digest(&body)),
    });
    let header = serde_json::to_vec(&r).unwrap();
    let parsed = wire::request(&mut frame(&header, &body).as_slice()).unwrap();
    assert_eq!(parsed.continuation_bytes, body);
    for mutation in ["version", "geometry", "length", "hash", "nan"] {
        let mut header = serde_json::to_value(&r).unwrap();
        let mut bytes = body.clone();
        match mutation {
            "version" => header["continuation"]["descriptor"]["schema_version"] = 2.into(),
            "geometry" => header["continuation"]["descriptor"]["shape"]["width"] = 416.into(),
            "length" => {
                bytes.pop();
            }
            "hash" => header["continuation"]["sha256"] = "b".repeat(64).into(),
            "nan" => {
                bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
                header["continuation"]["sha256"] =
                    format!("{:x}", sha2::Sha256::digest(&bytes)).into();
            }
            _ => unreachable!(),
        }
        let bytes = frame(&serde_json::to_vec(&header).unwrap(), &bytes);
        let mut called = false;
        assert_eq!(
            wire::serve(&mut bytes.as_slice(), &mut vec![], ready(&own), |_| {
                called = true;
                Ok(Never)
            }),
            Err(ErrorCode::InvalidRequest),
            "{mutation}"
        );
        assert!(!called, "{mutation}");
    }
}
#[test]
fn request_keeps_unicode_whitespace_and_exact_private_text() {
    let r = request("  synthetic åäö\n\ttext  ");
    let decoded = wire::request(&mut encoded(&r).as_slice())
        .unwrap_or_else(|_| panic!("rejected own CPU input"));
    assert_eq!(decoded.prompt, r.prompt);
    assert_eq!(decoded.seed, 42);
}
#[test]
fn actual_identity_is_canonical_verified_and_fails_on_changed_dependency() {
    let own = Own::new();
    let exe = std::env::current_exe().unwrap();
    let lib = own.0.join("cpu-dependency.txt");
    let digest = deployment::sha(&lib).unwrap();
    let map = BTreeMap::from([(lib.to_str().unwrap().into(), digest.clone())]);
    let identity = deployment::identity(&exe, &map, &[(&lib, &digest)], &[]).unwrap();
    assert_eq!(identity.worker_sha256, deployment::sha(&exe).unwrap());
    assert_eq!(
        identity.dependencies_sha256,
        format!(
            "{:x}",
            sha2::Sha256::digest(serde_json::to_vec(&map).unwrap())
        )
    );
    assert!(deployment::identity(&exe, &map, &[(&lib, &"0".repeat(64))], &[]).is_err());
    assert!(deployment::identity(&exe, &map, &[], std::slice::from_ref(&lib)).is_err());
    std::fs::write(&lib, b"changed own dependency").unwrap();
    assert!(deployment::identity(&exe, &map, &[], &[]).is_err());
}
#[test]
fn authoritative_export_rounding_and_invalid_pcm_are_closed() {
    assert_eq!(
        p::export_rgb8(&[-1., 0., 0.5, 1., 2.]).unwrap(),
        vec![0, 0, 128, 255, 255]
    );
    assert_eq!(
        p::export_rgb8(&[0.5_f32 / 255., 1.5_f32 / 255., 2.5_f32 / 255.]).unwrap(),
        vec![0, 2, 2]
    );
    assert!(p::export_rgb8(&[f32::NAN]).is_err());
    let mut s = Stream::new("cpu-request".into()).unwrap();
    let event = Event::AudioChunk {
        id: "cpu-request".into(),
        start: 0,
        samples: 1,
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
    bytes.extend_from_slice(&0f32.to_le_bytes());
    assert!(s.accept(&event, &bytes).is_err());
    assert!(
        s.accept(
            &Event::Complete {
                id: "cpu-request".into(),
                frames: 39,
                audio_samples: 52000
            },
            &[]
        )
        .is_err()
    );
}
#[test]
fn transport_write_failure_is_returned_without_exception_or_path_text() {
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("SYNTHETIC_PRIVATE_EXCEPTION"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert_eq!(
        wire::write(
            &mut Broken,
            &Event::Error {
                id: None,
                code: ErrorCode::GenerationFailed
            },
            &[]
        ),
        Err(ErrorCode::GenerationFailed)
    );
}
struct ChildGuard(Child);
impl std::ops::Deref for ChildGuard {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.0
    }
}
impl std::ops::DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.0.try_wait().is_ok_and(|s| s.is_none()) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}
fn spawn(own: &Own) -> ChildGuard {
    ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_h3-cpu-worker-fixture"))
            .env_clear()
            .env("H3_CPU_ROOT", &own.0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}
fn wait(child: &mut Child) -> std::process::ExitStatus {
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() > end {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CPU child exceeded bound")
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
#[test]
fn real_cpu_child_streams_complete_av_with_private_prompt_only_on_stdin() {
    let own = Own::new();
    let mut child = spawn(&own);
    let mut input = child.stdin.take().unwrap();
    let mut output = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let (ready, body) = event(&mut output);
    assert!(body.is_empty());
    let Event::Ready { deployment } = ready else {
        panic!("not ready")
    };
    assert_eq!(
        deployment.worker_sha256,
        deployment::sha(&PathBuf::from(env!("CARGO_BIN_EXE_h3-cpu-worker-fixture"))).unwrap()
    );
    let mut r = request("SYNTHETIC_PRIVATE_CANARY");
    r.profile = p::Profile::AvNoReferenceV2;
    r.shape = p::Shape {
        width: 64,
        height: 96,
        frames: 22,
        fps: 24,
    };
    input.write_all(&encoded(&r)).unwrap();
    drop(input);
    let mut stream = Stream::with_shape(r.id.clone(), r.shape).unwrap();
    loop {
        let (e, payload) = event(&mut output);
        stream.accept(&e, &payload).unwrap();
        assert!(!serde_json::to_string(&e).unwrap().contains(&r.prompt));
        if matches!(e, Event::Complete { .. }) {
            break;
        }
    }
    assert!(wait(&mut child).success());
    let mut tail = Vec::new();
    output.read_to_end(&mut tail).unwrap();
    assert!(tail.is_empty());
    let mut errors = String::new();
    stderr.read_to_string(&mut errors).unwrap();
    assert!(errors.is_empty());
    assert!(own.0.join("admitted").exists());
}
#[test]
fn missing_required_tail_never_publishes_complete_even_after_full_av() {
    let own = Own::new();
    let mut child = spawn(&own);
    let mut input = child.stdin.take().unwrap();
    let mut output = child.stdout.take().unwrap();
    assert!(matches!(event(&mut output).0, Event::Ready { .. }));
    let mut r = request("own missing tail fixture");
    r.profile = p::Profile::AvNoReferenceV2;
    r.capture_continuation = true;
    input.write_all(&encoded(&r)).unwrap();
    drop(input);
    loop {
        let (event, _) = event(&mut output);
        assert!(!matches!(event, Event::Complete { .. }));
        if let Event::Error { code, .. } = event {
            assert_eq!(code, ErrorCode::GenerationFailed);
            break;
        }
    }
    assert!(!wait(&mut child).success());
}
#[test]
fn cpu_child_partial_error_and_private_panic_never_publish_complete() {
    for prompt in [
        "SYNTHETIC_PARTIAL_FAILURE",
        "SYNTHETIC_PRIVATE_PANIC_CANARY",
    ] {
        let own = Own::new();
        let mut child = spawn(&own);
        let mut output = child.stdout.take().unwrap();
        assert!(matches!(event(&mut output).0, Event::Ready { .. }));
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&encoded(&request(prompt)))
            .unwrap();
        loop {
            let (e, _) = event(&mut output);
            assert!(!matches!(e, Event::Complete { .. }));
            if matches!(
                e,
                Event::Error {
                    code: ErrorCode::GenerationFailed,
                    ..
                }
            ) {
                break;
            }
        }
        assert!(!wait(&mut child).success());
        let mut errors = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut errors)
            .unwrap();
        assert!(errors.is_empty());
    }
}
#[test]
fn cpu_child_cancel_is_killed_and_reaped_without_any_gpu_or_ticket() {
    let own = Own::new();
    let mut child = spawn(&own);
    let mut output = child.stdout.take().unwrap();
    assert!(matches!(event(&mut output).0, Event::Ready { .. }));
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&encoded(&request("SYNTHETIC_WAIT")))
        .unwrap();
    let end = Instant::now() + Duration::from_secs(3);
    while !own.0.join("admitted").exists() {
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(child.try_wait().unwrap().is_none());
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
    assert!(child.try_wait().unwrap().is_some());
}
use sha2::Digest;

#[test]
fn i2v_body_requires_exact_rgb_dimensions_hash_and_profile_before_factory() {
    let own = Own::new();
    let mut r = request("synthetic i2v");
    r.profile = p::Profile::ImageToAvV1;
    r.image_rgb = vec![1, 2, 3, 4, 5, 6];
    r.image = Some(p::ImageInput {
        width: 2,
        height: 1,
        sha256: format!("{:x}", sha2::Sha256::digest(&r.image_rgb)),
    });
    let head = serde_json::to_vec(&r).unwrap();
    let good = frame(&head, &r.image_rgb);
    let decoded =
        wire::request(&mut good.as_slice()).unwrap_or_else(|_| panic!("valid i2v rejected"));
    assert_eq!(decoded.image_rgb, r.image_rgb);
    assert_eq!(decoded.profile, p::Profile::ImageToAvV1);
    let mut cases = vec![
        frame(&head, &[]),
        frame(&head, &[0; 6]),
        frame(&head, &[1; 7]),
    ];
    let mut bad = r.clone();
    bad.image.as_mut().unwrap().width = u32::MAX;
    cases.push(frame(&serde_json::to_vec(&bad).unwrap(), &[]));
    bad = r.clone();
    bad.image = None;
    cases.push(frame(&serde_json::to_vec(&bad).unwrap(), &[]));
    bad = r.clone();
    bad.profile = p::Profile::AvNoReferenceV2;
    let reference_frame0 =
        wire::request(&mut frame(&serde_json::to_vec(&bad).unwrap(), &r.image_rgb).as_slice())
            .unwrap();
    assert_eq!(reference_frame0.profile, p::Profile::AvNoReferenceV2);
    assert_eq!(reference_frame0.image_rgb, r.image_rgb);
    bad.profile = p::Profile::ShortAvNoReferenceV1;
    cases.push(frame(&serde_json::to_vec(&bad).unwrap(), &r.image_rgb));
    for input in cases {
        let mut called = false;
        let mut output = Vec::new();
        assert_eq!(
            wire::serve(&mut input.as_slice(), &mut output, ready(&own), |_| {
                called = true;
                Ok(Never)
            }),
            Err(ErrorCode::InvalidRequest)
        );
        assert!(!called);
    }
}

#[test]
fn startup_declaration_accepts_rebuild_but_rejects_missing_file_and_protocol_drift() {
    let own = Own::new();
    let exe = std::env::current_exe().unwrap();
    let lib = own.0.join("cpu-dependency.txt");
    let map = BTreeMap::from([(lib.to_str().unwrap().into(), deployment::sha(&lib).unwrap())]);
    let identity = deployment::identity(&exe, &map, &[], &[]).unwrap();
    std::fs::write(&lib, b"recompiled compatible dependency").unwrap();
    assert!(deployment::declaration(&exe, &map, identity.clone(), &[]).is_ok());
    let mut bad = identity.clone();
    bad.protocol += 1;
    assert!(deployment::declaration(&exe, &map, bad, &[]).is_err());
    std::fs::remove_file(&lib).unwrap();
    assert!(deployment::declaration(&exe, &map, identity, &[]).is_err());
}

#[test]
fn checkpoint_header_accepts_changed_weights_without_digest_and_rejects_bad_structure() {
    let own = Own::new();
    let path = own.0.join("own-model.safetensors");
    let header = serde_json::to_vec(
        &serde_json::json!({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
    )
    .unwrap();
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&[0; 8]).unwrap();
    drop(file);
    assert!(deployment::checkpoint(&path).is_ok());
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::End(-8)).unwrap();
    file.write_all(&[255; 8]).unwrap();
    drop(file);
    assert!(deployment::checkpoint(&path).is_ok());
    // A changed payload size or metadata does not require a lock digest update.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&[42; 16]).unwrap();
    drop(file);
    assert!(deployment::checkpoint(&path).is_ok());
    std::fs::write(&path, b"invalid header").unwrap();
    assert!(deployment::checkpoint(&path).is_err());
    assert!(deployment::checkpoint(&own.0).is_err());
    assert!(deployment::checkpoint(&own.0.join("missing.safetensors")).is_err());
}
