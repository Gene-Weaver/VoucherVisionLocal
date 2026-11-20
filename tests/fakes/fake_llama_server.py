"""A stand-in for llama.cpp's `llama-server`, for supervisor and end-to-end tests.

Accepts the flags VVL passes; behavior comes from environment variables:

  FAKE_LLAMA_MODE       comma-separated: ready (default), crash_on_start, crash_later,
                        ignore_sigterm, bind_fail_once, oom_on_start, no_devices,
                        selftest_glibc, chat_garbage, chat_500_once
  FAKE_LLAMA_LOAD_SECONDS   how long /health answers 503 "Loading model" (default 0.2)
  FAKE_LLAMA_CRASH_AFTER    seconds after ready before crash_later exits (default 0.5)
  FAKE_LLAMA_REPLY_DELAY    seconds each chat request takes (default 0)
  FAKE_LLAMA_STATE_DIR      folder for request logs and markers (requests.jsonl,
                            concurrency.txt, starts.txt, bind_failed)
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import sys
import threading
import time

MODE = set(filter(None, os.environ.get("FAKE_LLAMA_MODE", "ready").split(",")))
STATE = os.environ.get("FAKE_LLAMA_STATE_DIR") or "."
LOAD_SECONDS = float(os.environ.get("FAKE_LLAMA_LOAD_SECONDS", "0.2"))
CRASH_AFTER = float(os.environ.get("FAKE_LLAMA_CRASH_AFTER", "0.5"))
REPLY_DELAY = float(os.environ.get("FAKE_LLAMA_REPLY_DELAY", "0"))
API_KEY = os.environ.get("LLAMA_API_KEY")


def out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def arg(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def state_path(name):
    return os.path.join(STATE, name)


def main(argv):
    if "--version" in argv:
        if "selftest_glibc" in MODE:
            out("llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found "
                "(required by libllama.so.1)")
            return 1
        out("version: 1.0.0-dev (build 12359, commit abc1234)")
        out("built with fake for testing")
        return 0
    if "--list-devices" in argv:
        out("Available devices:")
        if "no_devices" not in MODE:
            out("  CUDA0: Fake GPU A (49140 MiB, 48000 MiB free)")
            out("  CUDA1: Fake GPU B (49140 MiB, 47000 MiB free)")
            out("  Vulkan0: Fake GPU A (49140 MiB, 48000 MiB free)")
            out("  Vulkan1: Fake GPU B (49140 MiB, 47000 MiB free)")
        return 0

    port = int(arg(argv, "--port", "0"))
    alias = arg(argv, "--alias", "fake")
    with open(state_path("starts.txt"), "a") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "argv": argv, "api_key_set": bool(API_KEY),
                                 "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "vk_visible": os.environ.get("GGML_VK_VISIBLE_DEVICES"),
                                 "llama_env": sorted(k for k in os.environ if k.startswith("LLAMA_"))}) + "\n")
    out(f"fake llama-server starting, model {arg(argv, '--model')}")
    if "ignore_sigterm" in MODE:
        signal.signal(signal.SIGTERM, lambda *_: out("fake: ignoring SIGTERM"))
    if "crash_on_start" in MODE:
        time.sleep(0.1)
        out("fake: GGML_ASSERT failed: something broke")
        return 3
    if "oom_on_start" in MODE:
        time.sleep(0.1)
        out("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 30000.00 MiB on device 0: "
            "cudaMalloc failed: out of memory")
        out("llama_model_load: error loading model: unable to allocate CUDA0 buffer")
        return 1
    if "bind_fail_once" in MODE and not os.path.exists(state_path("bind_failed")):
        open(state_path("bind_failed"), "w").close()
        out(f"start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: {port}")
        return 1

    started = time.monotonic()
    inflight = {"now": 0, "max": 0}
    lock = threading.Lock()
    flags = {"failed_once": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if time.monotonic() - started < LOAD_SECONDS:
                    return self._json(503, {"error": {"code": 503, "message": "Loading model",
                                                      "type": "unavailable_error"}})
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"data": [{"id": alias}]})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if API_KEY and self.headers.get("Authorization") != f"Bearer {API_KEY}":
                return self._json(401, {"error": {"code": 401, "message": "Invalid API Key",
                                                  "type": "authentication_error"}})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            with lock:
                inflight["now"] += 1
                inflight["max"] = max(inflight["max"], inflight["now"])
                with open(state_path("concurrency.txt"), "w") as handle:
                    handle.write(str(inflight["max"]))
            try:
                with open(state_path("requests.jsonl"), "a") as handle:
                    handle.write(json.dumps(body) + "\n")
                if REPLY_DELAY:
                    time.sleep(REPLY_DELAY)
                if "chat_500_once" in MODE and not flags["failed_once"]:
                    flags["failed_once"] = True
                    return self._json(503, {"error": {"code": 503, "message": "busy"}})
                fmt = body.get("response_format") or {}
                schema = (fmt.get("json_schema") or {}).get("schema") if isinstance(fmt, dict) else None
                if "chat_garbage" in MODE:
                    content = "Sorry, I cannot produce JSON for this."
                elif schema:
                    content = json.dumps({name: f"fake {name}" for name in schema.get("properties", {})})
                else:
                    content = "FAKE OCR TEXT\nJouvea pilosa\nMICH"
                return self._json(200, {
                    "id": "chatcmpl-fake", "object": "chat.completion", "model": alias,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290},
                })
            finally:
                with lock:
                    inflight["now"] -= 1

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    out(f"main: server is listening on http://127.0.0.1:{port}")
    if "crash_later" in MODE:
        def _crash():
            time.sleep(LOAD_SECONDS + CRASH_AFTER)
            out("fake: CUDA error: an illegal memory access was encountered")
            os._exit(7)
        threading.Thread(target=_crash, daemon=True).start()

    def _term(*_):
        out("fake: shutting down")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    if "ignore_sigterm" not in MODE:
        signal.signal(signal.SIGTERM, _term)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
