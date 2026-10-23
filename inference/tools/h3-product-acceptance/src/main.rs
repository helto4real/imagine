//! Explicit synthetic GPU acceptance; one canonical Ticket and one native child.
use anyhow::{Context, Result, bail};
use imagine_h3_product_acceptance::{coordination::Ticket, deployment, protocol as p};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, process::Stdio, time::Instant};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command,
    time::{Duration, timeout},
};
const SOCKET: &str = "/home/thhel/ChatGPT/imagine/.imagine/gpu.sock";
async fn event(reader: &mut (impl AsyncRead + Unpin)) -> Result<(p::Event, Vec<u8>)> {
    async fn part(reader: &mut (impl AsyncRead + Unpin), limit: usize) -> Result<Vec<u8>> {
        let n = reader.read_u32().await.context("event_length_failed")? as usize;
        if n > limit {
            bail!("event_limit_failed")
        }
        let mut bytes = vec![0; n];
        reader
            .read_exact(&mut bytes)
            .await
            .context("event_read_failed")?;
        Ok(bytes)
    }
    let header = part(reader, p::MAX_HEADER).await?;
    let event = serde_json::from_slice(&header).context("event_header_failed")?;
    Ok((event, part(reader, p::MAX_PAYLOAD).await?))
}
// Never drop the Ticket while direct-child exit remains unconfirmed.
async fn reap(child: &mut tokio::process::Child) -> std::process::ExitStatus {
    loop {
        let _ = child.start_kill();
        if let Ok(status) = child.wait().await {
            return status;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().context("own_root_required")?).canonicalize()?;
    let case = args
        .next()
        .and_then(|v| v.into_string().ok())
        .context("case_required")?;
    if args.next().is_some() || !matches!(case.as_str(), "baseline" | "alternate" | "cancel") {
        bail!("invalid_arguments")
    }
    if !root.starts_with(
        "/home/thhel/.codex/worktrees/h3-product-worker/imagine/comfyui/h3-product-20261020",
    ) {
        bail!("invalid_own_root")
    }
    let run = root.join(format!("{}-{}", case, std::process::id()));
    std::fs::create_dir(&run)?;
    let executable = root.join("bin/imagine-h3-worker");
    let environment: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(root.join("manifests/gpu-environment.json"))?)?;
    let map_path = PathBuf::from(&environment[deployment::MAP_ENV]);
    let dependencies: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(&map_path)?)?;
    let kitchen = PathBuf::from(&environment["IMAGINE_KITCHEN_LIBRARY"]);
    let aimdo = PathBuf::from(&environment["IMAGINE_AIMDO_LIBRARY"]);
    let identity = deployment::identity(
        &executable,
        &dependencies,
        &[
            (&kitchen, deployment::BRIDGE_SHA256),
            (
                &aimdo,
                &dependencies[aimdo.to_str().context("invalid_anchor")?],
            ),
        ],
        &[map_path],
    )
    .map_err(|_| anyhow::anyhow!("deployment_mismatch"))?;
    let profile: Value = serde_json::from_str(include_str!("../../../profiles/h3-short-av.json"))?;
    let prompt = if case == "alternate" {
        "A polished red wooden sphere rolls gently across a grey stone surface in a quiet room."
            .to_owned()
    } else {
        profile["synthetic_prompt"]
            .as_str()
            .context("source_fixture_missing")?
            .to_owned()
    };
    let request = p::Request {
        protocol: p::VERSION,
        kind: p::RequestKind::GenerateVideo,
        id: format!("h3-product-{case}-{}", std::process::id()),
        profile: p::Profile::ShortAvNoReferenceV1,
        prompt,
        seed: 42,
    };
    if !request.valid() {
        bail!("invalid_synthetic_request")
    }
    let header = serde_json::to_vec(&request)?;
    let started = Instant::now();
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut ticket = timeout(Duration::from_secs(120), async {
        Ticket::register(std::path::Path::new(SOCKET), &request.id, header.len())
            .await?
            .acquire()
            .await
    })
    .await
    .context("grant_timeout")??;
    let mut command = Command::new(&executable);
    command
        .env_clear()
        .envs(&environment)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .current_dir(&run);
    let parent = std::process::id() as libc::pid_t;
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 || libc::getppid() != parent
            {
                return Err(std::io::Error::other("parent_exit"));
            }
            Ok(())
        });
    }
    let mut child = match command.spawn() {
        Ok(v) => v,
        Err(_) => {
            let _ = ticket.finish().await;
            bail!("worker_spawn_failed")
        }
    };
    let pid = child.id().context("worker_pid_missing")?;
    let mut input = child.stdin.take().context("stdin_missing")?;
    let mut output = BufReader::new(child.stdout.take().context("stdout_missing")?);
    let mut events = Vec::<Value>::new();
    let mut video = Sha256::new();
    let mut audio = Sha256::new();
    let mut frames = 0;
    let mut samples = 0;
    let mut completed = false;
    let mut cancelled = false;
    let flow = async {
        let (ready, body) = event(&mut output).await?;
        if !body.is_empty() || !matches!(ready,p::Event::Ready{deployment} if deployment==identity)
        {
            bail!("ready_mismatch")
        }
        events.push(json!({"kind":"ready","seconds":started.elapsed().as_secs_f64()}));
        input.write_u32(header.len() as u32).await?;
        input.write_all(&header).await?;
        input.write_u32(0).await?;
        input.flush().await?;
        drop(input);
        let mut stream = p::Stream::new(request.id.clone()).context("invalid_id")?;
        loop {
            let (event, payload) = event(&mut output).await?;
            stream
                .accept(&event, &payload)
                .map_err(anyhow::Error::msg)?;
            match event {
                p::Event::Phase { phase, .. } => {
                    events.push(json!({"kind":"phase","phase":phase,"seconds":started.elapsed().as_secs_f64()}));
                    println!("{} phase {}", case, serde_json::to_string(&phase)?);
                    if case == "cancel" && matches!(phase, p::Phase::EncodingText) {
                        cancelled = true;
                        break;
                    }
                }
                p::Event::Progress {
                    completed,
                    total,
                    elapsed_seconds,
                    ..
                } => {
                    events.push(json!({"kind":"progress","completed":completed,"total":total,"elapsed_seconds":elapsed_seconds}));
                    println!("{case} sampling {completed}/{total}");
                }
                p::Event::VideoFrame { .. } => {
                    video.update(&payload);
                    frames += 1;
                }
                p::Event::AudioChunk { samples: count, .. } => {
                    audio.update(&payload);
                    samples += count;
                }
                p::Event::Complete { .. } => {
                    completed = true;
                    break;
                }
                p::Event::Error { code, .. } => {
                    events.push(json!({"kind":"error","code":code}));
                    bail!("closed_worker_error")
                }
                p::Event::Ready { .. } => bail!("duplicate_ready"),
                p::Event::ReferenceCache { .. } => {
                    bail!("unexpected_reference_cache_in_no_reference_probe")
                }
            }
        }
        Ok::<_, anyhow::Error>(())
    };
    let result = tokio::select! {
        result=timeout(Duration::from_secs(1800),flow)=>result.map_err(|_|anyhow::anyhow!("worker_timeout")).and_then(|r|r),
        _=ticket.lost()=>Err(anyhow::anyhow!("ticket_lost")),
        _=tokio::signal::ctrl_c()=>Err(anyhow::anyhow!("controller_cancelled")),
        _=terminate.recv()=>Err(anyhow::anyhow!("controller_cancelled")),
    };
    let mut forced = result.is_err() || cancelled;
    let status = if forced {
        reap(&mut child).await
    } else {
        match timeout(Duration::from_secs(10), child.wait()).await {
            Ok(Ok(status)) => status,
            _ => {
                forced = true;
                reap(&mut child).await
            }
        }
    };
    let reaped = !std::path::Path::new(&format!("/proc/{pid}")).exists();
    let finish_sent = ticket.finish().await.is_ok();
    let passed = result.is_ok()
        && reaped
        && finish_sent
        && if case == "cancel" {
            cancelled && !completed && forced
        } else {
            completed
                && status.success()
                && !forced
                && frames == p::FRAMES
                && samples == p::AUDIO_SAMPLES
        };
    let record = json!({"schema":1,"case":case,"result":if passed{"passed"}else{"failed"},"worker":identity,"ticket_socket":SOCKET,"worker_pid":pid,"child_reaped":reaped,"finish_sent_after_reap":finish_sent,"forced_stop":forced,"exit_code":status.code(),"complete":completed,"frames":frames,"audio_samples":samples,"video_rgb8_sha256":format!("{:x}",video.finalize()),"pcm_f32le_sha256":format!("{:x}",audio.finalize()),"elapsed_seconds":started.elapsed().as_secs_f64(),"events":events,"error":result.err().map(|_|"acceptance_flow_failed"),"prompt_source":if case=="alternate"{"own alternate source fixture"}else{"frozen source profile fixture"},"media_retained":false});
    std::fs::write(
        run.join("acceptance.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    println!(
        "{} {} {}",
        case,
        if passed { "PASS" } else { "FAIL" },
        run.display()
    );
    if !passed {
        bail!("acceptance_failed")
    }
    Ok(())
}
