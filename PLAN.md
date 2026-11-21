# VoucherVisionLocal plan

VoucherVisionLocal (VVL) runs the VoucherVision pipeline on the user's own
computer: label detection, OCR, and parsing into structured records. Models run
through a managed llama.cpp server or through the user's own API keys.

## Decisions

- Not a fork of VoucherVisionGO. Working code was copied once (see `UPSTREAM.md`)
  and the two projects may diverge.
- No LLM server fork. VVL manages a pinned, checksummed llama.cpp `llama-server`
  release and talks to it over the OpenAI-compatible API.
- Desktop app from the start: a thin Electron shell around a Flask backend that
  also serves the UI, so the same UI works in a browser.
- Python dependencies are managed and locked with uv. No torch, transformers, or
  langchain.
- TextCollage's label detector is LeafMachine3's archival detector (YOLO26x),
  downloaded at first start: Core ML on macOS, onnxruntime on the CPU elsewhere.
  No OpenVINO.
- All keys live only in the gitignored `PRIVATE.yaml`.
- Model and engine files live outside the repo (`models_dir` in settings).
- Enrichment helpers (elevation, World Flora Online) go through the hosted
  VoucherVisionGO API; WFO can also be an optional local database download.
- Local models come from a verified registry. Starting families: Qwen3-VL,
  Muse, MiMo, GLM, Mistral, Kimi, Gemma 4. Qwen3-VL is the first test case.
  Nothing is downloaded without an explicit confirmation.

## Tabs

1. **VVL** — the processing page, modeled on the VoucherVisionGO web page.
2. **Models** — local models and API models, clearly separated.
3. **Keys** — one card per credential, each with its purpose.
4. **Prompts** — built-in and user prompt templates.
5. **Engine** — system report and llama.cpp management.

## Phases

| Phase | Content | Status |
|---|---|---|
| 1 | GUI, backend skeleton, keys, prompts, system checker, demo engine | built 2026-11-19 (415 tests pass; Linux only, not yet run on Windows or macOS) |
| 2 | Managed llama.cpp engine, one Qwen3-VL download, real local processing, model-fit recommendations | built 2026-11-19 (570 tests pass; verified on Linux with the Vulkan build and Qwen3-VL 32B Q8; CUDA builds, Windows, and macOS not yet run) |
| 2b | Engine pool: a model on every graphics card, OCR and parsing models side by side when they fit, two-pass batches when they do not; header resource monitor; sticky bars | built 2026-11-19 (verified on Linux with Qwen3-VL 2B, 8B, and 32B) |
| 3 | API providers: OpenRouter (replaces Hyperbolic; one key for Claude, GPT, Gemini, Mistral, and others), then native Gemini, Claude, and OpenAI adapters; enrichment through VVGO including a new `/wfo` route, optional local WFO database | OpenRouter backend built 2026-11-20 (docs/API.md section 14; not yet run with a real key); model gallery layout being chosen (scratch_designs/openrouter_gallery); the rest not started |
| 4 | Remaining verified model families, one at a time | not started |
| 5 | Installers with a bundled uv, CI on Windows/macOS/Linux, auto-update | not started |

The backend/UI contract is `docs/API.md`.
