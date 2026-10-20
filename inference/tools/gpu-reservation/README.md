# Native GPU reservation development runner

This standalone local tool shares the product's exact `src/coordination.rs`
Ticket implementation. Build with its own Cargo home/target and `--locked
--offline`. `--bytes` is the existing protocol's job cost, not a VRAM limit.

```
imagine-native-gpu-reservation --socket /absolute/private/gpu.sock \
  --job own-synthetic-id --bytes 1 --timeout 120 \
  --manifest /absolute/own-output/reservation.json -- /absolute/native-worker
```

In the default process form, the worker must be a direct native ELF process with **all GPU work in that
single process**, with no GPU children or session changes. Interpreter/shell/
isolation launchers are rejected. This tool does not supervise arbitrary process
trees. In particular, do not pass `test_isolation.py` or bwrap as its worker.
The separate explicit `--process-form bwrap-pid-namespace` mode permits a
reviewed Python/reference worker inside `/usr/bin/bwrap`. Supply
`--namespace-entry /absolute/own-build/namespace_entry`, reviewed bwrap mounts,
then `--` and the worker command after the controller's argument separator.
The normal direct-worker contract is unchanged.

This mode creates a PID namespace and opens its init pidfd before admitting
the worker. A small GPU-free native entry helper waits on a private, newly
created 0700 directory/socket mounted by the controller. A one-use random token
and peer UID check identify its connection. Only an explicit `G` byte admits
execution; EOF, wrong byte or setup failure cannot start the reviewed payload.
The caller supplies the filesystem/device/network mounts and must review them
for each workload. This is lifecycle supervision, not an adversarial sandbox.
Supervisor fd/namespace options and arguments-file overrides are rejected.

Cancellation, timeout, lease loss and launcher completion stop the namespace
init. The controller retains the Ticket until both the init pidfd confirms exit
and the launcher has been reaped. Kernel PID-namespace cleanup includes detached
session children and double-fork descendants. It does not provide controller
SIGKILL/fatal-crash tolerance: the caller must await cleanup, as with the direct
runner. A launcher failure before publishing its init PID is recorded as an
unconfirmed namespace exit; the admission helper still cannot receive a grant.

The controller waits for a grant before starting the worker. Cancellation,
SIGINT/SIGTERM, timeout or lost coordination kills and reaps its own direct
worker before releasing its Ticket. Never abandon the `run` future before that
cleanup finishes. `finish_sent` records a send, not broker acknowledgement.
No uncoordinated fallback is supported.

Seventeen CPU tests passed: six direct fake-listener lifecycle/worker-boundary
cases, two direct launcher signal cases, six namespace lifecycle cases and
three strict-entry admission cases. Signal cases send SIGINT and SIGTERM to their
own controller and verify that their known harmless worker PID has disappeared
before accepting `finished`. Namespace cases also verify that known detached
and double-fork peers disappear before `finished`; admission cases verify
EOF/wrong-byte denial and explicit-byte success. All-target warnings-denied Clippy passed. These
tests contact only their own temporary sockets; they perform no GPU work and
make no claim of a trained inference run or fault-tolerant arbitrary supervision.
