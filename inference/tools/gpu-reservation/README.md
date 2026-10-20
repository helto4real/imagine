# Native GPU reservation development runner

This standalone local tool shares the product's exact `src/coordination.rs`
Ticket implementation. Build with its own Cargo home/target and `--locked
--offline`. `--bytes` is the existing protocol's job cost, not a VRAM limit.

```
imagine-native-gpu-reservation --socket /absolute/private/gpu.sock \
  --job own-synthetic-id --bytes 1 --timeout 120 \
  --manifest /absolute/own-output/reservation.json -- /absolute/native-worker
```

The worker must be a direct native ELF process with **all GPU work in that
single process**, with no GPU children or session changes. Interpreter/shell/
isolation launchers are rejected. This tool does not supervise arbitrary process
trees. In particular, do not pass `test_isolation.py` or bwrap as its worker.
An isolation wrapper may instead enclose this controller and its direct worker,
with the explicitly authorized coordinator socket mounted inside that boundary.
That composition remains a separate gate before any such GPU validation.

The controller waits for a grant before starting the worker. Cancellation,
SIGINT/SIGTERM, timeout or lost coordination kills and reaps its own direct
worker before releasing its Ticket. Never abandon the `run` future before that
cleanup finishes. `finish_sent` records a send, not broker acknowledgement.
No uncoordinated fallback is supported.

Eight CPU tests passed: six fake-listener lifecycle/worker-boundary cases and
two actual launcher signal cases. The latter send SIGINT and SIGTERM to their
own controller and verify that their known harmless worker PID has disappeared
before accepting `finished`. All-target warnings-denied Clippy passed. These
tests contact only their own temporary sockets; they perform no GPU work and
make no claim of a trained inference run or arbitrary descendant cleanup.
