# Independent H3 worker GPU acceptance

This developer-only CPU supervisor drives the real, unchanged product worker
under one Ticket from the existing canonical coordinator. It is not the product
API/queue/UI wiring and does not replace that integration or its tests.

Direct human approval in this chat on 2026-10-20 lifted the GPU pause. The same
approval covered product integration; automatic permission review nevertheless
rejected the attempted product-source transfer. That denied action did not run,
and no product files or alternative product wiring are changed here.

Build with the task-local Cargo home/target and `--locked --offline`. Packages
are an exact subset of the frozen inference/reservation locks. This executable
links no inference/Torch library; only its admitted direct child runs CUDA.

The command takes only its own fixed root and one source-fixture case:

```
imagine-h3-product-acceptance /absolute/own/h3-product-20261020 baseline
imagine-h3-product-acceptance /absolute/own/h3-product-20261020 alternate
imagine-h3-product-acceptance /absolute/own/h3-product-20261020 cancel
```

The root contains `bin/imagine-h3-worker`, a separately prepared dependency map
and `manifests/gpu-environment.json`. Those are own deployment/build artifacts,
not ordinary product configuration. Startup hashes the actual child executable,
all dependencies and the frozen bridge before registering/acquiring a Ticket.
It never starts a coordinator or falls back to uncoordinated execution.
The worker revalidates the same map before Ready, then checks all four locked
models before trained allocations. Requests and media use only private pipes.

`baseline` uses the committed profile's synthetic blue-cube prompt. `alternate`
uses an explicit source-owned red-sphere prompt, retaining every locked setting.
No private prompt, prompt file, model input, AV byte or exception text is saved.
The supervisor validates Ready, every bounded event, final frame/sample counts,
finite PCM and complete four-step progress with the authoritative shared Stream.
It retains metadata and streaming RGB8/PCM digests only, in a fresh case/PID
subdirectory. Error/timeout/cancellation/coordination loss kill and reap the
single owned child before the Ticket finish send. `cancel` stops after the
EncodingText event, which the worker emits after loading the trained text graph.
A subsequent complete run is needed to demonstrate recovery.

These results establish actual worker execution/transport/lifecycle for the
explicit fixtures. They do not establish intermediate tensor equality against
the frozen oracle, peak GPU memory, native AV playback, common-scheduler image
recovery, UI Save, or the blocked product integration. A finish send is not a
broker acknowledgement. Controller SIGKILL/fatal-crash tolerance is not claimed.
