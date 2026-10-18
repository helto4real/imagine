"""Synthetic child for Rust lifecycle tests; never imports the GPU adapter."""
import argparse
import os
import struct
import sys
import time
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import serve


def png(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x20\x40\x60" * width) * height)) + chunk(b"IEND", b""))


class FakeRuntime:
    def generate(self, header, references, emit):
        emit("phase", phase="inference")
        mode = header["prompt"]
        if mode == "hang":
            time.sleep(60)
        if mode == "disconnect":
            os._exit(7)
        if mode == "failure":
            raise RuntimeError("synthetic-private-sentinel")
        if mode == "wrong_id":
            emit("phase", phase="inference", id="wrong")
        if mode == "invalid_output":
            return b"not a PNG"
        if mode == "adapter":
            # Exercise the real adapter against the Rust-produced request, with
            # tiny stdlib doubles for tensors/loaders instead of GPU imports.
            from test_worker import RuntimeTests
            runtime, _, _ = RuntimeTests.generation_runtime()
            runtime.Image.fromarray.return_value.save.side_effect = lambda output, **kw: output.write(png(header["width"], header["height"]))
            return runtime.generate(header, references, emit)
        emit("progress", completed=header["sampling"]["steps"], total=header["sampling"]["steps"], elapsed_seconds=1.0)
        return png(header["width"], header["height"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.parse_args()
    serve(sys.stdin.buffer, sys.stdout.buffer, FakeRuntime())
