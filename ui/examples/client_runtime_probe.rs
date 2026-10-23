//! Own loopback-only experiment. No Imagine service, models, files or GPU.
use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
#[path = "../src/async_runtime.rs"]
mod async_runtime;

fn serve(mut socket: TcpStream, stop: Arc<AtomicBool>, requests: Arc<AtomicUsize>) {
    socket
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    let mut reader = BufReader::new(socket.try_clone().unwrap());
    while !stop.load(Ordering::Acquire) {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(_) => break,
            Ok(_) => {}
        }
        if !line.starts_with("GET /own-probe HTTP/1.1") {
            break;
        }
        loop {
            line.clear();
            if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
                break;
            }
        }
        requests.fetch_add(1, Ordering::AcqRel);
        if socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nContent-Type: application/json\r\nConnection: keep-alive\r\n\r\n{\"ok\":true}").is_err() { break; }
    }
}
fn main() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/own-probe", listener.local_addr().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let requests = Arc::new(AtomicUsize::new(0));
    let connections = Arc::new(AtomicUsize::new(0));
    let (flag, seen, sockets) = (stop.clone(), requests.clone(), connections.clone());
    let server = std::thread::spawn(move || {
        let mut workers = vec![];
        while !flag.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((socket, _)) => {
                    sockets.fetch_add(1, Ordering::AcqRel);
                    let (flag, seen) = (flag.clone(), seen.clone());
                    workers.push(std::thread::spawn(move || serve(socket, flag, seen)));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(e) => panic!("own listener: {e}"),
            }
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let mut phases = vec![];
    for mode in ["ephemeral", "persistent", "ephemeral"] {
        // Same shared reqwest client and per-request current-thread runtime pattern
        // as the UI; persistent is the only controlled variant.
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(750))
            .build()
            .unwrap();
        let runtime = (mode == "persistent").then(async_runtime::runtime);
        let start = Instant::now();
        let before = (
            requests.load(Ordering::Acquire),
            connections.load(Ordering::Acquire),
        );
        let mut workers = vec![];
        for _ in 0..8 {
            let (client, url) = (client.clone(), url.clone());
            workers.push(std::thread::spawn(move || {
                let mut failures = vec![];
                for i in 0..40 {
                    let request = async {
                        let response = client.get(&url).send().await?;
                        response.json::<serde_json::Value>().await
                    };
                    let result = if let Some(runtime) = &runtime { runtime.block_on(request) }
                        else { tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(request) };
                    match result {
                        Ok(value) => assert_eq!(value, serde_json::json!({"ok":true})),
                        Err(e) => {
                            let mut chain = vec![];
                            let mut source = std::error::Error::source(&e);
                            while let Some(cause) = source { chain.push(cause.to_string()); source = cause.source(); }
                            failures.push(serde_json::json!({"iteration":i,"timeout":e.is_timeout(),"connect":e.is_connect(),"request":e.is_request(),"body":e.is_body(),"decode":e.is_decode(),"causes":chain}));
                        }
                    }
                }
                failures
            }));
        }
        let failures = workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect::<Vec<_>>();
        phases.push(serde_json::json!({"mode":mode,"attempts":320,"failures":failures.len(),"examples":failures.into_iter().take(5).collect::<Vec<_>>(),"seconds":start.elapsed().as_secs_f64(),"requests_received":requests.load(Ordering::Acquire)-before.0,"connections_accepted":connections.load(Ordering::Acquire)-before.1}));
        drop(client);
    }
    stop.store(true, Ordering::Release);
    server.join().unwrap();
    println!(
        "{}",
        serde_json::json!({"experiment":"shared-client/runtime-lifetime A/B/A","workers":8,"requests_per_worker":40,"test_timeout_ms":750,"server":"own std::net loopback keepalive responder","persistent_variant":"actual candidate async_runtime.rs","product_services_started":false,"gpu":false,"phases":phases,"server_joined":true})
    );
}
