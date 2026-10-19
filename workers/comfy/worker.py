"""Private framed IPC. Only main() imports the GPU runtime; tests use a fake runtime."""
import json
import os
import struct
import sys

PROTOCOL = 1
ATTENTION = "comfy_kitchen_int8"
MAX_HEADER = 1024 * 1024
MAX_PAYLOAD = 256 * 1024 * 1024


class WorkerError(Exception):
    def __init__(self, code, diagnostic=None):
        self.code = code
        self.diagnostic = diagnostic


def component_load(component, operation):
    """Report only a bounded cause and optional tensor key from the loader."""
    try:
        return operation()
    except WorkerError:
        raise
    except Exception as error:
        text = str(error).lower()
        tensor = None
        if isinstance(error, FileNotFoundError):
            reason = "file_not_found"
        elif isinstance(error, PermissionError):
            reason = "permission_denied"
        elif isinstance(error, MemoryError) or "outofmemory" in type(error).__name__.lower() or "out of memory" in text:
            reason = "out_of_memory"
        elif isinstance(error, KeyError):
            reason = "missing_tensor"
            key = error.args[0] if error.args else None
            if isinstance(key, str) and 0 < len(key) <= 200 and all(c.isascii() and (c.isalnum() or c in "_.") for c in key):
                tensor = key
        elif "size mismatch" in text or "shape mismatch" in text:
            reason = "tensor_shape_mismatch"
        elif "quantization" in text or "unsupported dtype" in text:
            reason = "unsupported_quantization"
        elif "could not detect" in text or "vae is invalid" in text:
            reason = "unsupported_architecture"
        elif isinstance(error, NotImplementedError) or "unsupported format" in text:
            reason = "unsupported_format"
        elif "safetensor" in type(error).__name__.lower() or "header" in text or isinstance(error, (ValueError, EOFError)):
            reason = "invalid_checkpoint"
        else:
            reason = "load_failed"
        diagnostic = {"component": component, "reason": reason}
        if tensor is not None:
            diagnostic["tensor"] = tensor
        raise WorkerError("model_component_load_failed", diagnostic) from None


def exact(reader, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = reader.read(count - len(chunks))
        if not chunk:
            raise EOFError()
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(reader):
    size = struct.unpack(">I", exact(reader, 4))[0]
    if not 0 < size <= MAX_HEADER:
        raise WorkerError("comfy_protocol_error")
    header = json.loads(exact(reader, size))
    size = struct.unpack(">I", exact(reader, 4))[0]
    if size > MAX_PAYLOAD:
        raise WorkerError("comfy_protocol_error")
    return header, exact(reader, size)


def write_frame(writer, header, payload=b""):
    raw = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise WorkerError("comfy_protocol_error")
    writer.write(struct.pack(">I", len(raw)) + raw + struct.pack(">I", len(payload)))
    writer.write(payload)
    writer.flush()


def references(header, payload):
    lengths = header.get("reference_lengths", [])
    if (not isinstance(lengths, list) or len(lengths) > 10
            or any(type(n) is not int or n <= 0 for n in lengths)
            or sum(lengths) != len(payload)):
        raise WorkerError("comfy_protocol_error")
    result = []
    offset = 0
    for length in lengths:
        result.append(payload[offset:offset + length])
        offset += length
    return result


def serve(reader, writer, runtime):
    write_frame(writer, {"type": "ready", "protocol": PROTOCOL, "attention": ATTENTION})
    while True:
        try:
            header, payload = read_frame(reader)
        except EOFError:
            return
        request_id = header.get("id")
        if (header.get("protocol") != PROTOCOL or header.get("type") != "generate"
                or not isinstance(request_id, str) or not 0 < len(request_id) <= 64):
            raise WorkerError("comfy_protocol_error")

        def emit(event, **values):
            write_frame(writer, {"type": event, "id": request_id, **values})

        try:
            result = runtime.generate(header, references(header, payload), emit)
            write_frame(writer, {"type": "result", "id": request_id}, result)
        except Exception as error:
            # Never serialize raw exception messages, paths, prompts or tracebacks.
            code = error.code if isinstance(error, WorkerError) else "comfy_generation_failed"
            event = {"type": "error", "id": request_id, "code": code}
            if isinstance(error, WorkerError) and error.diagnostic is not None:
                event["diagnostic"] = error.diagnostic
            write_frame(writer, event)
            return  # A failed generation discards all GPU/model state.


def main():
    # Reserve a private output descriptor before importing noisy third party code.
    writer = os.fdopen(os.dup(1), "wb", buffering=0)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    sys.dont_write_bytecode = True
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--comfy-root", required=True)
        args = parser.parse_args()
        root = os.path.abspath(args.comfy_root)
        # -I excludes even this script's directory from sys.path.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.modules["worker"] = sys.modules[__name__]
        from runtime import Runtime
        runtime = Runtime(root)
        serve(sys.stdin.buffer, writer, runtime)
    except Exception as error:
        code = error.code if isinstance(error, WorkerError) else "comfy_startup_failed"
        write_frame(writer, {"type": "error", "id": None, "code": code})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
