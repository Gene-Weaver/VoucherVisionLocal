# VoucherVisionLocal backend API contract (Phase 1)

This is the agreement between the Flask backend (`app.py` + `vvl/`), the web UI
(`ui/`), and the Electron shell (`desktop/`). If something here turns out to be
wrong or missing, do not silently diverge: implement the closest sensible thing
and list the difference in your final report so this file can be updated.

## 1. Process model

- One Python process: Flask app object `app` in `app.py`, served by waitress.
  Started with `uv run vvl serve [--port N] [--host 127.0.0.1]`.
- It serves the UI and the API from the same origin. Default port **8642**.
- The Electron shell spawns that command, waits for `GET /health`, then loads
  `http://127.0.0.1:<port>/?token=<token>`. The same URL works in a browser.

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `VVL_PORT` | Listen port | `8642` |
| `VVL_TOKEN` | Session token; generated at startup if unset | random |
| `VVL_OWNER_PID` | If set, the server exits when this process dies | unset |
| `VVL_HOME` | Runtime data folder (logs, cache, temp) | OS user-data dir `VoucherVisionLocal` |
| `VVL_PRIVATE_PATH` | Credentials file | `<repo>/PRIVATE.yaml` from source, else `<VVL_HOME>/PRIVATE.yaml` |
| `VVL_SETTINGS_PATH` | Settings file | `<repo>/settings.local.yaml` from source, else `<VVL_HOME>/settings.yaml` |

"From source" means a `pyproject.toml` sits beside `app.py`.

`vvl serve` prints one line containing the full URL with `?token=` so a browser
user can click it. It must not log the token anywhere else.

## 2. Security rules (all mandatory)

1. Bind to `127.0.0.1` only unless `--host` is given explicitly.
2. Reject any request whose `Host` header is not `127.0.0.1[:port]`,
   `localhost[:port]`, or `[::1][:port]` with HTTP 403 (DNS-rebinding defense).
3. Every route except the static UI files and `GET /health` requires the session
   token, sent as `X-VVL-Token: <token>`. `Authorization: Bearer <token>` and
   `X-API-Key: <token>` are accepted as aliases (the second keeps the
   VoucherVisionGO client package usable). Missing or wrong token: HTTP 401.
   Compare with `hmac.compare_digest`. Fail closed if no token is configured.
4. The token is never embedded in any HTML, JS, or JSON the server sends. The UI
   reads it from the `?token=` query string, keeps it in `sessionStorage`, and
   removes it from the address bar with `history.replaceState`.
5. No permissive CORS. Send no `Access-Control-Allow-Origin` header at all.
6. Secret values from `PRIVATE.yaml` are write-only: no route ever returns one,
   and they are never logged. The most a route returns is `saved: true` and a
   mask (`"••••" + last 4 characters`, or just `"••••"` when shorter than 12).
7. File paths supplied by a client are confined: prompt names must match
   `^[A-Za-z0-9._-]+\.ya?ml$` and resolve inside the prompts folders;
   `models_dir` must be an absolute path. No route takes an arbitrary file path
   to read.
8. UI code never puts untrusted text into `innerHTML` without escaping.
9. Uploads are size-capped and handled through temp files that are always removed.

## 3. Error envelope

Every error response from every route uses the envelope produced by
`_error_payload()` in VoucherVisionGO's `app.py` (kept as-is in the copy). The
UI already knows how to display it.

## 4. Model identifiers

A model id is a string with a source prefix:

| Form | Example | Meaning |
|---|---|---|
| `demo:mock` | `demo:mock` | Built-in demo engine. Sample output, no model. |
| `local:<build_id>` | `local:qwen3-vl-8b-q8` | A build from `vvl/registry/models_local.yaml` |
| `api:<provider_id>:<model>` | `api:gemini:gemini-3.1-flash-lite` | A provider model |

In Phase 1 only `demo:mock` is runnable. Everything else is listed with
`runnable: false` and a human-readable `unavailable_reason`.

## 5. Routes kept from VoucherVisionGO (same request and response shapes)

Borrow the implementation from the copied `app.py`. The request field names and
the response JSON are unchanged, with two differences: `engines` and `llm_model`
carry VVL model ids, and all Gemini/Vertex/quota/billing fields are gone.

| Route | Notes |
|---|---|
| `GET /health` | Public. Adds `"app": "VoucherVisionLocal"`, `"version"`, `"pid"`. |
| `POST /process` | Multipart image upload. |
| `POST /process-url` | Image by URL (keeps the guarded/anti-bot fetchers). |
| `GET /prompts` | Same listing shape VVGO returns, built-in + user prompts. |
| `GET /api-costs` | Contents of `vvl/registry/api_cost.yaml`. |

PDF processing: the VVGO async job system (Cloud Tasks, GCS, email) is removed.
Keep the synchronous in-process PDF path (`process_pdf_request`) reachable
through `/process` when a PDF is uploaded.

### Demo engine behavior

With `demo:mock`, the request runs every real step that needs no model
(validation, resize, TextCollage label detection, prompt loading) and then:

- OCR text is a fixed sample transcription.
- The parsed record has exactly the fields of the selected prompt. Values are
  taken from the sample where a field name matches, else empty strings.
- Tokens and costs are zero.
- The response carries a top-level `"engine_notice"` string saying this is demo
  output and not a real transcription. The UI shows it prominently.

A request naming a non-runnable model returns HTTP 409 with code
`MODEL_NOT_AVAILABLE` and the model's `unavailable_reason`.

## 6. New routes (`/v1/...`)

All return JSON. All require the token.

### `GET /v1/app`

```json
{
  "name": "VoucherVisionLocal",
  "version": "0.1.0",
  "phase": 1,
  "from_source": true,
  "python": "3.12.15",
  "packaged": false,
  "paths": {
    "home": "/home/u/.local/share/VoucherVisionLocal",
    "models_dir": "/datac/VoucherVisionLocal_Models",
    "private_file": "/path/to/PRIVATE.yaml",
    "settings_file": "/path/to/settings.local.yaml",
    "prompts_builtin": "/path/to/prompts/builtin",
    "prompts_user": "/path/to/prompts/user"
  },
  "features": {
    "demo_engine": true,
    "local_inference": false,
    "api_inference": false,
    "engine_install": false,
    "model_download": false,
    "enrichment_elevation": true,
    "enrichment_wfo": false
  },
  "feature_notes": {
    "local_inference": "Local models arrive in Phase 2.",
    "api_inference": "API providers arrive in Phase 3.",
    "enrichment_wfo": "World Flora Online lookups arrive in Phase 3."
  }
}
```

`paths.private_file` is the path only. The file is never read for this route.
`python` is the version of the Python running the server
(`platform.python_version()`). `packaged` is true when the environment variable
`VVL_PACKAGED` is `"1"`, which the desktop shell sets in the installed app.

### `GET /v1/system` and `GET /v1/system?refresh=1`

Returns the system report produced by `vvl.system.get_report(refresh: bool)`.
Cached in memory; `refresh=1` re-probes.

```json
{
  "schema": 1,
  "generated_at": "2026-11-19T18:04:11Z",
  "duration_ms": 412,
  "os": {
    "system": "Linux", "name": "Ubuntu 22.04.4 LTS", "version": "22.04",
    "kernel": "6.8.0-138-generic", "arch": "x86_64",
    "is_wsl": false, "is_apple_silicon": false, "python": "3.12.15"
  },
  "cpu": {
    "model": "AMD Ryzen Threadripper PRO 5975WX", "vendor": "AuthenticAMD",
    "physical_cores": 32, "logical_cores": 64, "max_mhz": 4500,
    "features": {"avx": true, "avx2": true, "avx512": false, "fma": true, "f16c": true, "neon": false}
  },
  "memory": {
    "total_bytes": 270000000000, "available_bytes": 200000000000,
    "swap_total_bytes": 8000000000, "swap_free_bytes": 8000000000
  },
  "gpus": [
    {
      "index": 0, "vendor": "nvidia", "name": "NVIDIA RTX 6000 Ada Generation",
      "kind": "discrete",
      "vram_total_bytes": 51527024640, "vram_free_bytes": 50000000000,
      "driver_version": "550.120", "backend_hints": ["cuda", "vulkan"],
      "source": "nvidia-smi", "details": {}
    }
  ],
  "accelerators": {
    "cuda":   {"available": true,  "detail": "driver 550.120, CUDA 12.4"},
    "metal":  {"available": false, "detail": "macOS only"},
    "vulkan": {"available": null,  "detail": "vulkaninfo not found"}
  },
  "disk": {
    "models_dir": "/datac/VoucherVisionLocal_Models", "exists": false,
    "checked_path": "/datac", "total_bytes": 8000000000000, "free_bytes": 3000000000000,
    "is_synced_folder": false, "synced_folder_hint": null
  },
  "capability": {
    "recommended_backend": "cuda",
    "backend_reason": "NVIDIA GPU with a working driver was found.",
    "gpu_memory_budget_bytes": 96000000000,
    "cpu_memory_budget_bytes": 200000000000,
    "tier": "workstation",
    "summary": "2x NVIDIA RTX 6000 Ada (48 GB each), 64 threads, 251 GB RAM"
  },
  "warnings": [{"code": "MODELS_DIR_MISSING", "message": "The models folder does not exist yet. It is created on first download."}],
  "probes": [{"name": "gpu_nvidia", "ok": true, "duration_ms": 38, "error": null}]
}
```

- `kind`: `discrete | integrated | unified` (`unified` = Apple Silicon).
- `vendor`: `nvidia | amd | intel | apple | unknown`.
- `available: null` means "could not be determined".
- `tier`: `cpu_only | entry | mid | high | workstation`.
- Any field may be `null` when a probe fails. The report is always returned.

Model fit, used by `/v1/models`:
`vvl.system.fit.assess(report: dict, build: dict) -> dict`

```json
{"verdict": "fits_gpu", "label": "Fits on GPU", "required_bytes": 11200000000,
 "detail": "Needs about 11 GB; 48 GB of GPU memory is free on GPU 0."}
```

`verdict`: `fits_gpu | fits_split | cpu_only | too_large | unknown`.
`build` is one `builds[]` entry from `models_local.yaml`.

### `GET /v1/providers`

One entry per provider in `vvl/registry/providers.yaml` (enabled ones only).

```json
{"providers": [{
  "id": "gemini", "label": "Google Gemini", "role": "model_provider", "kind": "gemini",
  "purpose": "…", "sends_images": true, "optional": true,
  "key_url": "https://…", "docs_url": "https://…",
  "fields": [{"name": "api_key", "label": "Gemini API key", "placeholder": "…",
              "saved": true, "masked": "••••a1b2"}],
  "configured": true,
  "last_test": {"ok": true, "status": "ok", "http_status": 200,
                "message": "Key accepted.", "tested_at": "2026-11-19T18:04:11Z"}
}]}
```

`configured` is true when every field has a saved value. `last_test` is `null`
until a test runs in this session (kept in memory only).

### `PUT /v1/keys/<provider_id>`

Body `{"fields": {"api_key": "…"}}`. An empty string removes that field. Writes
`PRIVATE.yaml` atomically (temp file + rename), with file mode `0600` where the
OS supports it, preserving entries for other providers. Returns the provider
object as in `GET /v1/providers`. Clears `last_test` for that provider.

### `DELETE /v1/keys/<provider_id>`

Removes all saved fields for that provider. Returns the provider object.

### `POST /v1/keys/<provider_id>/test`

Runs the provider's `test` request from the registry with the saved key and a
10-second timeout.

```json
{"ok": false, "status": "invalid_key", "http_status": 401,
 "message": "The service rejected this key.", "tested_at": "…"}
```

`status`: `ok | invalid_key | not_configured | network_error | error`.
The response never echoes the key or the provider's raw response body.

### `GET /v1/models`

```json
{
  "local": {"families": [{
    "id": "qwen3-vl", "label": "Qwen3-VL", "developer": "Alibaba Qwen",
    "license": "apache-2.0", "status": "verified", "roles": ["ocr", "parse"],
    "summary": "…", "homepage": "https://…",
    "builds": [{
      "id": "qwen3-vl-8b-q8", "model_id": "local:qwen3-vl-8b-q8",
      "label": "Qwen3-VL 8B (Q8)", "params": "8B", "quant": "Q8_0", "tier": "medium",
      "hf_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
      "total_size_bytes": 9460000000,
      "files": [{"role": "model", "path": "…", "size_bytes": 8710000000}],
      "install_state": "not_installed",
      "fit": {"verdict": "fits_gpu", "label": "Fits on GPU", "required_bytes": 0, "detail": "…"},
      "runnable": false,
      "unavailable_reason": "Local models arrive in Phase 2."
    }]
  }]},
  "api": {"providers": [{
    "id": "gemini", "label": "Google Gemini", "kind": "gemini", "sends_images": true,
    "configured": false,
    "models_state": "no_key",
    "models_error": null,
    "models": [{"id": "gemini-…", "model_id": "api:gemini:gemini-…", "label": "gemini-…"}],
    "runnable": false,
    "unavailable_reason": "API providers arrive in Phase 3."
  }]},
  "demo": [{"model_id": "demo:mock", "label": "Demo engine (sample output, no model)",
            "roles": ["ocr", "parse"], "runnable": true}],
  "selection": {"ocr": ["demo:mock"], "parse": "demo:mock"},
  "selectable": {
    "ocr":   [{"model_id": "demo:mock", "label": "Demo engine", "group": "Demo",
               "provider_label": null, "runnable": true, "unavailable_reason": null,
               "capabilities": {"vision": true, "thinking_levels": [], "structured_output": true}}],
    "parse": []
  }
}
```

- `install_state`: `not_installed | downloading | installed | error`
  (always `not_installed` in Phase 1).
- `models_state`: `no_key | not_loaded | loaded | error`.
- Only providers with `role: model_provider` appear under `api`.
- `selectable` is the flat list the VVL tab's pickers render, in display order,
  with `group` one of `Demo | Local | API`. It includes non-runnable entries so
  the user can see what is coming; the UI disables them and shows the reason.
- `capabilities.thinking_levels` is the list of levels the model accepts (empty
  list = hide the thinking-level control).
- If `vvl.system` is missing or raises, `fit` is
  `{"verdict": "unknown", "label": "Not checked", "required_bytes": null, "detail": "…"}`.

### `POST /v1/providers/<provider_id>/models/refresh`

Fetches the provider's model list using the saved key (registry `models` spec),
caches it in memory, and returns that provider's object from `api.providers`.
This is the only route that lists remote models; nothing calls a provider
automatically.

### `PUT /v1/selection`

Body `{"ocr": ["demo:mock"], "parse": "demo:mock"}`. Validates the ids exist,
saves to settings, returns `{"selection": {...}}`.

### `GET /v1/settings`, `PUT /v1/settings`

```json
{"models_dir": "/datac/VoucherVisionLocal_Models",
 "selection": {"ocr": ["demo:mock"], "parse": "demo:mock"},
 "enrichment": {"elevation": true, "wfo_source": "off"},
 "ui": {"theme": "dark"},
 "setup": {"usage": null}}
```

`setup.usage` is the answer to the Setup page's "How will you run models?":
`openrouter | local | both`, or `null` until the user chooses (the UI opens the
Setup page on first run while it is `null`).

`PUT` accepts a partial object with these keys only; unknown keys are rejected
with HTTP 400. `wfo_source`: `off | vvgo | local`. The response to `PUT` also
carries `"warnings": []`, e.g. when `models_dir` is inside a synced folder.

### Prompts

`GET /v1/prompts`

```json
{"prompts": [{
  "name": "SLTPvM_default.yaml", "source": "builtin",
  "prompt_name": "…", "prompt_author": "…", "prompt_author_institution": "…",
  "prompt_version": "…", "prompt_description": "…", "llm": "…",
  "n_fields": 30, "valid": true, "error": null
}]}
```

`GET /v1/prompts/<source>/<name>` (`source` = `builtin | user`)

```json
{"name": "…", "source": "builtin", "prompt_name": "…", "prompt_author": "…",
 "prompt_author_institution": "…", "prompt_version": "…", "prompt_description": "…",
 "llm": "…", "instructions": "…", "json_formatting_instructions": "…",
 "fields": [{"name": "catalogNumber", "rule": "…", "group": "TAXONOMY"}],
 "mapping": {"TAXONOMY": ["…"]},
 "yaml_text": "…", "valid": true, "error": null}
```

`POST /v1/prompts/import` — multipart `file`, or JSON
`{"name": "x.yaml", "yaml_text": "…"}`. Validated with the same rules VVGO
applies to user prompt uploads. Saved to the user prompts folder. Returns the
prompt summary object. A name that collides with a built-in prompt is rejected.

`DELETE /v1/prompts/user/<name>` — removes a user prompt. Built-ins cannot be
deleted.

### Engine

`GET /v1/engine`

```json
{"available": false,
 "unavailable_reason": "The managed llama.cpp engine arrives in Phase 2.",
 "managed": {"installed": false, "version": null, "pinned_version": null,
             "running": false, "pid": null, "port": null,
             "backend": null, "loaded_model": null},
 "models_dir": "/datac/VoucherVisionLocal_Models"}
```

`POST /v1/engine/install`, `/v1/engine/start`, `/v1/engine/stop` return HTTP 501
with code `NOT_IMPLEMENTED` in Phase 1.

### `POST /v1/shutdown`

Replies `{"ok": true}`, then exits the process hard from a daemon thread
(`os._exit`). Used by the desktop shell on quit.

## 7. Static UI

- `GET /` returns `ui/index.html`.
- `GET /ui/<path>` serves files under `ui/` (so pages reference `ui/css/...`,
  `ui/js/...`). No CDN or other external assets: everything is vendored under
  `ui/vendor/`.
- Static responses carry `Cache-Control: no-store` so a restarted app never
  shows stale code.

## 8. As implemented (backend complete, 2026-11-19)

These points refine the sections above. Where they differ, this section wins.

### Error envelope

Flat object, as in VoucherVisionGO:

```json
{"error": "Model 'local:x' is not available.", "error_code": "MODEL_NOT_AVAILABLE",
 "message": "Local models arrive in Phase 2.", "host": "VoucherVisionLocal",
 "request_id": "…", "details": {"model_id": "…", "unavailable_reason": "…"}}
```

Codes added for VVL: `MODEL_NOT_AVAILABLE` (409), `NOT_IMPLEMENTED` (501),
`NOT_FOUND`, `METHOD_NOT_ALLOWED`, `PAYLOAD_TOO_LARGE` (413), `CONFLICT` (409),
`FORBIDDEN` (403). Billing, Vertex, quota, and approval codes are gone.
A malformed or unknown model id is HTTP 400 `INVALID_INPUT` with
`details.model_id`.

### `/process`, `/process-url`

- Missing `engines` / `llm_model` fall back to the saved selection. The default
  prompt is `SLTPvM_full.yaml`.
- Uploads are capped at 50 MB (413). Unknown extensions are 415; an unreadable
  image is 400.
- Result JSON: `payment_inference` is removed; `impact` is kept and is `null`;
  `engine_notice` is the last key (string for the demo engine, else `null`; the
  PDF wrapper has it too); `ocr_info` is keyed by VVL model id; `WFO_info` is
  always `""` until Phase 3; `success.text_collage` reports whether a collage
  was actually built (when no labels are found, the whole image is used).

### `/prompts` (legacy listing)

JSON errors use the envelope (`PROMPT_NOT_FOUND` with
`details.available_prompts`). `format=text` errors are plain text.

### Prompts (`/v1/prompts...`)

- Import returns HTTP 201 with the summary object. A name that collides with an
  existing user prompt or a built-in (case-insensitive) is 409. Files over
  256 KiB are 413. Names starting with `.` are rejected. A multipart upload may
  send a `name` field that overrides the file name.
- Delete of a user prompt returns `{"ok": true, "name": "…", "source": "user"}`.
  Deleting a built-in is 403 `FORBIDDEN`.
- `yaml_text` always uses `\n` line endings. A field's `group` is `null` when
  the mapping does not list it.

### `/v1/settings`

- `ui.theme`: `light | dark | system` (default `dark`).
- `warnings` is a list of `{"code": "…", "message": "…"}` with codes
  `MODELS_DIR_SYNCED` and `MODELS_DIR_MISSING` (the second only when
  `models_dir` changed in that request).
- Unknown keys are rejected at nested levels too (`details.unknown_keys`).
  `~` in `models_dir` is expanded.

### `/v1/selection`

A partial body is allowed. `ocr` holds at most 4 ids, no repeats.

### `/v1/models`

- `selectable.parse` is populated the same way as `selectable.ocr`.
- A provider's models appear in `selectable` only after its list has been
  refreshed in this session.
- For local entries, `provider_label` is the family label.

### `POST /v1/providers/<id>/models/refresh`

With no key saved: HTTP 200, `models_state: "no_key"`, no network call. A failed
fetch gives `models_state: "error"` and `models_error`. Providers that are not
model providers (vvgo, huggingface) are 400. Only the first page of a
provider's list is read.

### Keys

- `PUT /v1/keys/<id>`: the body may contain only `fields`. Unknown field names
  or non-string values are 400. Values are trimmed; the limit is 8192 characters.
- Key test: other HTTP codes give `status: "error"` with the message
  "The service answered with HTTP N."

### `/v1/system` fallback

If the system checker is missing or fails, the same top-level shape is returned
with null fields, `gpus: []`, and a warning with code `SYSTEM_CHECK_UNAVAILABLE`.

### Server

- Extra response headers on everything: `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`,
  `Cache-Control: no-store`, `X-Request-ID`.
- `vvl serve --threads N` (default 8). One stdout line:
  `VoucherVisionLocal <version> is running. Open: http://127.0.0.1:<port>/?token=…`
- Extra environment variable: `VVL_PROMPTS_USER_DIR` (user prompt folder), for
  tests and scratch runs.

### System report additions (`vvl/system`)

All fields in section 6 are present. Extra fields:

- `os`: `python_arch`, `is_translated`, `wsl_version`, `wsl_gpu_passthrough`.
- `cpu.features`: `avx_vnni`, `avx512_vnni`, `amx`, `sve`, `dotprod`, `i8mm`; and
  `cpu.features_source`.
- `accelerators.cuda`: `state`, `driver_version`, `cuda_version`.
- `accelerators.vulkan.devices[]`.
- `disk`: `models_dir_source`, `writable`, `filesystem`, `is_network`.
- `capability`: `memory_mode` (`discrete | shared | cpu`), `gpu_indices`.
- `warnings[].severity`: `info | warning`.

Notes:

- `capability.gpu_memory_budget_bytes` is capacity minus a reserve, not current
  free memory, so verdicts do not flicker with other programs' usage.
- On unified-memory machines `gpus[].vram_total_bytes` is system RAM.
- Fit labels: "Fits on GPU", "Split across GPUs", "Partly on GPU",
  "CPU only (slow)", "Too large", "Unknown".
- The memory estimate and tier thresholds live in `vvl/system/policy.py`.
- `python -m vvl.system [--summary] [--fit] [--models-dir PATH]` prints the report.

## 9. Phase 2: managed llama.cpp engine and local models

Phase 2 makes `local:<build_id>` models runnable. `GET /v1/app` then reports
`features.local_inference`, `features.engine_install`, and
`features.model_download` as `true`.

### Storage

Everything large lives under `models_dir` (never in the repo or `VVL_HOME`):

- `<models_dir>/engine/<release tag>/<variant id>/` — the extracted engine.
- `<models_dir>/models/<hf org>/<hf repo>/<file>` — model files. Two builds that
  share a file (for example one vision projector) share it on disk.
- Downloads in progress use a `.partial` suffix and are resumable.

The engine release and every downloadable file are pinned with sha256 in
`vvl/llama/engine.lock.yaml` and `vvl/registry/models_local.yaml`. Nothing is
used until its sha256 has been verified.

### Jobs (downloads and installs)

Long operations run as background jobs. The UI polls about once a second while
any job is active.

`GET /v1/jobs` → `{"jobs": [Job]}` (active jobs plus recently finished ones).
`GET /v1/jobs/<id>` → `Job`.

```json
{"id": "j_ab12", "kind": "model_download", "target": "qwen3-vl-32b-q8",
 "label": "Qwen3-VL 32B (Q8)",
 "state": "running",
 "bytes_done": 1200000000, "bytes_total": 35590000000,
 "speed_bps": 85000000, "eta_seconds": 404,
 "current_file": "Qwen3VL-32B-Instruct-Q8_0.gguf",
 "files": [{"path": "…", "size_bytes": 0, "bytes_done": 0, "state": "running"}],
 "error": null,
 "started_at": "…", "finished_at": null}
```

- `kind`: `model_download | engine_install | detector_download | detector_gpu_download` (section 15).
- `state`: `queued | running | verifying | done | error | canceled`.
- `files[].state`: `pending | running | verifying | done | skipped | error`
  (`skipped` = already present and verified).
- `error`: `{"code": "…", "message": "…"}` with codes such as `NETWORK_ERROR`,
  `CHECKSUM_MISMATCH`, `INSUFFICIENT_DISK_SPACE`, `HTTP_ERROR`, `GATED_MODEL`
  (the model needs a Hugging Face token / accepted license), `EXTRACT_FAILED`.

### Local model downloads

`POST /v1/models/local/<build_id>/download`
Body `{"confirm_size_bytes": <total_size_bytes of the build>}`. The number must
equal the build's `total_size_bytes`; this proves the user was shown the size.
Returns HTTP 202 `{"job": Job}`.
Errors: 400 `INVALID_INPUT` (missing or wrong confirmation), 404 unknown build,
409 `CONFLICT` (already installed, or a download of it is already running),
507 `INSUFFICIENT_DISK_SPACE` with `details.required_bytes` and
`details.free_bytes` (required = remaining bytes plus a 2 GB margin).
Only one model download runs at a time; a second request while another build
is downloading is 409 with `details.active_job_id`.

`POST /v1/models/local/<build_id>/download/cancel` → `{"job": Job}`. Keeps the
partial file so a later download resumes.

`DELETE /v1/models/local/<build_id>` → `{"ok": true, "freed_bytes": N}`. Removes
the build's files (and partials) that no other installed build shares. 409 if
the build is currently loaded in the engine.

`GET /v1/models` build objects gain:

```json
{"install_state": "installed", "installed_bytes": 35590000000,
 "download_job_id": null, "recommended": true,
 "runnable": true, "unavailable_reason": null}
```

- `install_state`: `not_installed | partial | downloading | installed | error`
  (`partial` = an interrupted download can be resumed).
- `recommended`: at most one build per family, the most capable build whose fit
  verdict is `fits_gpu` (or the best lower verdict when none fits a GPU).
- A build is `runnable` when it is installed and the engine is installed.
  Otherwise `unavailable_reason` says which step is missing.

### Engine

`GET /v1/engine`

```json
{"available": true, "unavailable_reason": null,
 "models_dir": "/datac/VoucherVisionLocal_Models",
 "managed": {
   "pinned_version": "b12359", "installed": true, "version": "b12359",
   "variant": {"id": "linux-x64-cuda12", "label": "Linux x64, NVIDIA CUDA 12", "backend": "cuda"},
   "state": "ready",
   "running": true, "pid": 1234, "port": 8671, "backend": "cuda",
   "loaded_model": "local:qwen3-vl-32b-q8",
   "gpu_devices": [1], "context_size": 16384, "parallel": 2,
   "started_at": "…", "last_error": null,
   "idle_stop_at": "…"
 },
 "recommended_variant": {"id": "linux-x64-cuda12", "label": "…", "backend": "cuda",
                         "download_size_bytes": 793600000, "reason": "…"},
 "variants": [{"id": "…", "label": "…", "backend": "…", "download_size_bytes": 0,
               "compatible": true, "reason": null, "installed": false}],
 "install_job_id": null,
 "settings": {"variant": "auto", "context_size": "auto", "parallel": 2,
              "gpu_devices": "auto", "idle_stop_minutes": 15}}
```

- `state`: `not_installed | stopped | starting | loading_model | ready | stopping | error`.
- `variants` lists only builds for this OS and CPU type.
- `settings.gpu_devices`: `"auto"` or a list of GPU indexes from `/v1/system`.
  Auto uses the single GPU with the most free memory when the model fits on it,
  otherwise all GPUs.
- `settings.idle_stop_minutes`: the engine is stopped after this long without a
  request, to free graphics memory. `0` disables it.

`PUT /v1/settings` accepts an `engine` object with those five keys.

`POST /v1/engine/install` — body `{"variant": "<id>"}` optional (default: the
recommended variant). 202 `{"job": Job}`. After extraction the installer runs
the engine's version and device-list commands; if a GPU build cannot start on
this machine the job fails with `ENGINE_SELF_TEST_FAILED` and the message names
the next variant to try.

`POST /v1/engine/uninstall` → engine object. 409 while running.

`POST /v1/engine/start` — body `{"model_id": "local:<build_id>"}`. Returns 202
with the engine object (`state: "starting"`); poll `GET /v1/engine` until
`ready` or `error`. Starting a different model while one is loaded restarts the
engine with the new model.

`POST /v1/engine/stop` → engine object.

`GET /v1/engine/logs?after=<seq>&limit=<n>` →
`{"lines": [{"seq": 12, "ts": "…", "text": "…"}], "next": 13}`. A ring buffer of
recent engine output (the engine's API key is never logged).

### Processing with a local model

`/process` and `/process-url` accept `local:<build_id>` for `engines` and
`llm_model`. If the engine is not running with that model it is started (or
switched) first; the request waits for it, up to a load timeout, and fails with
503 `ENGINE_NOT_READY` otherwise. If the OCR and parsing models differ, the
engine switches between them for each step (slow; the UI warns).

- OCR: one chat request with the image and the OCR prompt.
- Parsing: one chat request with the parsing prompt and a JSON schema built from
  the prompt's fields, so the reply is always valid JSON with exactly those
  fields.
- Token counts come from the engine's usage report; costs are zero.
- Errors from the engine use code `ENGINE_ERROR` (502) with a readable message.

The engine listens on 127.0.0.1 only, on a port chosen at start, protected by a
per-start random API key. It is stopped when VVL exits, and a stale engine left
by a crashed VVL is cleaned up at the next start.

## 10. File names from URLs, and live resource use

Both routes need the session token, like every `/v1` route.

### `POST /v1/url-names`

The file name VoucherVisionLocal gives an image URL. The UI uses it wherever a
name is needed before an image has been fetched (the single-URL run, a batch
URL list, in one request per list) and never works a name out of a URL itself.

Body: `{"urls": ["https://…", …]}` — at most 5000 URLs, each at most 4096
characters. An empty list is allowed.

```json
{"names": [{"url": "https://quod.lib.umich.edu/cgi/i/image/api/image/herb00ic:1122841:MICH-V-1122841/full/res:0/0/native.jpg",
            "name": "MICH-V-1122841.jpg", "stem": "MICH-V-1122841"}]}
```

- One entry per URL, in the order sent (duplicates included). `stem` is `name`
  without its extension.
- Names come from `vvl/fetch/url_name_parser.py`
  (`resolve_filename_from_response(url)` with no response metadata). This is
  the request-free core of `extract_filename_from_url`: pure string parsing,
  so the route never contacts the URL's server. (`extract_filename_from_url`
  itself may send a HEAD request, which is why it is not called here.)
- Text that is not a usable URL gets the parser's own fallback name
  (`image_<hash>.jpg`); it does not fail the batch.
- Errors: 400 `INVALID_INPUT` when the body is not JSON, `urls` is missing or
  not a list, an entry is not text (`details.index`), a URL is too long
  (`details.index`, `details.max_url_length`), there are too many URLs
  (`details.max_urls`, `details.count`), or the body has other keys
  (`details.unknown_keys`).
- After an image has been fetched, the `filename` in the `/process-url` result
  is the name to use: it comes from the same parser, which can then also use
  the final URL and the server's `Content-Disposition` header. The UI switches
  to it as soon as a result arrives (result label, history entry, download
  names, batch rows, zip entry names).

### `GET /v1/metrics`

Live use of the processor, memory and each graphics card, for the monitor in
the page header. Cheap and safe to call every 2 seconds: hardware readings are
cached for 1.5 s, so any number of clients cause one reading; while one reading
is in progress other callers get the previous one. A value that cannot be
measured is `null` (the UI shows "—"); a failed reading never turns into an
error response.

```json
{"ts": "2026-11-19T22:18:52Z",
 "cpu": {"percent": 4.8, "logical_cores": 64},
 "memory": {"total_bytes": 540653092864, "used_bytes": 78580436992,
            "available_bytes": 462072655872, "percent": 14.5},
 "gpus": [{"index": 1, "name": "NVIDIA RTX 6000 Ada Generation", "vendor": "nvidia", "kind": "discrete",
           "util_percent": 0, "temperature_c": 43,
           "vram_total_bytes": 51527024640, "vram_used_bytes": 27262976, "vram_free_bytes": 51499761664,
           "engine_bytes": 0, "other_bytes": 27262976,
           "planned_bytes": 41495660608,
           "planned_for": [{"model_id": "local:qwen3-vl-32b-q8", "label": "Qwen3-VL 32B (Q8)", "bytes": 41495660608}],
           "planned_fits": true,
           "memory_source": "nvidia-smi"}],
 "engine": {"state": "stopped", "loaded_model": null, "gpu_devices": null},
 "plan": {"mode": "single", "note": "One local model is used, loaded once.",
          "models": [{"model_id": "local:qwen3-vl-32b-q8", "label": "Qwen3-VL 32B (Q8)", "roles": ["ocr", "parse"],
                      "required_bytes": 41495660608, "gpu_indices": [1], "loaded": false}]}}
```

- `cpu.percent`: use since the previous reading (psutil, never a blocking
  measurement). `null` until there is a previous reading to compare with, so a
  meaningless first `0.0` is never reported.
- `memory`: `used_bytes` is `total_bytes - available_bytes`.
- `gpus[]`: one entry per card of `GET /v1/system`, with the same `index`.
  - `vram_free_bytes` is `vram_total_bytes - vram_used_bytes`.
  - `engine_bytes`: memory held by VVL's own engine process on this card;
    `other_bytes`: the rest of `vram_used_bytes`. `engine_bytes` is `0` when the
    engine is not running (or was started on other cards), and `null` when it
    runs here but its share cannot be measured; `other_bytes` is then `null` too.
  - `planned_bytes`: memory that will be set aside on this card for the selected
    local models and is not loaded yet; `planned_for` lists the models it is
    for. `planned_fits` is `false` when `planned_bytes` is more than
    `vram_free_bytes`, `null` when free memory is unknown.
  - `memory_source`: `nvidia-smi | sysfs | system_ram | null`.
- `engine`: `state` and `loaded_model` as in `GET /v1/engine`; `gpu_devices` is
  the list the running engine was started on.
- `plan.mode`: `none` (no local model selected) | `single` | `alternating`.
  `plan.models[]` has one entry per selected local model: its roles, its
  estimated need, the cards it goes to, and whether it is loaded now.
  Demo and API models need no graphics memory and are not listed.

Where the numbers come from:

| Hardware | Load | Memory | Engine's share |
|---|---|---|---|
| NVIDIA | `nvidia-smi --query-gpu` (also temperature) | same call | `nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid`, matched to the engine's process id |
| AMD on Linux | sysfs `gpu_busy_percent` (temperature from the card's hwmon) | sysfs `mem_info_vram_used` / `mem_info_vram_total` | not measurable (`null` while the engine runs) |
| Unified memory (Apple Silicon; NVIDIA GB10 class) | `null` (NVIDIA: from `nvidia-smi` when it reports it) | the `memory` figures (system RAM) | not measurable |
| Anything else | `null` | total from the system report if known, used `null` | not measurable |

`nvidia-smi` runs through `vvl/system/_proc.py` with a 1.5 s limit per call.
After a failure it is not called again for 15 s (120 s when it is not
installed); the cards are still listed, with `null` readings. On a computer
with no graphics card `gpus` is `[]`.

The plan (`vvl/llama/plan.py`, `memory_plan()`), which mirrors what the engine
does today:

- The models are the saved `selection` (`ocr` and `parse`).
- A model's need and its cards come from `EngineSupervisor.launch_preview()`,
  which uses the supervisor's own code: the engine settings (`context_size`,
  `parallel`), its memory estimate (file sizes + context × parallel × KV bytes
  per token + a runtime margin; a little above the `fit` estimate shown on the
  Models tab), and its GPU choice (`choose_gpu_devices()`: the
  `engine.gpu_devices` setting, or automatically the card with the most free
  memory when the model fits on it, otherwise all cards). Free memory is the
  live figure, plus what the engine holds now, because a start stops the loaded
  model first. A model spread over several cards is divided in proportion to
  the room on each.
- The engine holds one model at a time. With two different local models
  (`mode: "alternating"`) they are loaded in turn, so `planned_bytes` on a card
  is the larger of their shares, not the sum; `plan.note` says so.
- A model that is loaded is in `engine_bytes`, not in `planned_bytes`. When a
  different model is loaded, only what the waiting model needs beyond the
  engine's current memory on that card is planned.
- A CPU engine build plans no graphics memory.

## 11. Several engines and two-pass batches

The managed engine is a pool of llama-server engines (`vvl/llama/pool.py`).
Which model runs on which graphics card, with how many request slots, is
decided by `vvl/llama/planner.py` from the selected models and the memory each
card can spare (free memory minus 2 GB, counting our own engines as free):

- **One model**: an engine on every card it fits on, each with up to 4 request
  slots. Never two copies of a model on one card (measured: copies on the same
  card add nothing; slots do).
- **Two models that fit together** (`shared`): both stay loaded, on every card
  with room. No request waits for a model swap.
- **Two models that do not fit together** (`exclusive`): only the model needed
  now is loaded, on every card. Batches then run in two passes.
- A model too large for any one card gets a single engine spread over all cards.

Engines that already serve a wanted model are never restarted because the plan
shifts slightly; new engines are placed in the memory that is really free.

### Settings (`engine`)

- `parallel`: `"auto"` (default) or 1..16 request slots per engine.
- `staged_batches`: `false` (default). When `true`, batches always run in two
  passes. They do so regardless when the layout is `exclusive`.

### `GET /v1/engine` additions

```json
{"managed": {
   "engines": [{"number": 1, "model_id": "local:qwen3-vl-8b-q8", "state": "ready",
                "gpu_devices": [0], "parallel": 4, "context_size": 8192,
                "active_requests": 0, "pid": 1, "port": 8660, "started_at": "…"}],
   "loaded_models": ["local:qwen3-vl-8b-q8"],
   "parallel_total": 8,
   "layout": {"mode": "single", "waiting": []}
 },
 "plan": {"mode": "single", "staged": false, "concurrency": 8, "fits": true,
          "summary": "Qwen3-VL 8B (Q8): 2 copies, 8 images at a time."}}
```

- `managed.*` single-engine fields (`loaded_model`, `pid`, `port`, `parallel`,
  ...) describe the first engine; `gpu_devices` is the union over all engines;
  `active_requests` is the total.
- `layout.mode`: `none | single | shared | exclusive` for what is running;
  `layout.waiting` lists selected models that are not loaded right now.
- `plan` is what the pool will run for the saved selection, whether or not
  anything is running: `staged` tells a batch to use two passes, `concurrency`
  is how many images a batch should send at once.

`GET /v1/metrics` → `plan.mode` uses the same values (`alternating` is gone).

### Two-pass requests

`POST /process` and `POST /process-url` accept `stage=ocr`. The request stops
after OCR, saves the result, and returns:

```json
{"staged": true, "stage": "ocr", "stage_token": "…", "filename": "…",
 "url_source": "", "host": "VoucherVisionLocal", "ocr_info": {…}, "ocr": "…",
 "engine_notice": null}
```

An OCR-only or notebook request ignores `stage` and returns its normal result.
PDFs are always processed in one pass.

`POST /v1/process/finish` — JSON or form body with `stage_token`, plus the
usual optional `llm_model`, `prompt`, `include_wfo`, `include_cop90`,
`llm_thinking_level`. Parses the saved OCR text and returns exactly what a
one-pass `/process` returns. The saved result is removed when the image is
finished; an unknown, used, or expired token (3 days) is 404
`STAGE_NOT_FOUND`. Saved results live under `<VVL_HOME>/cache/stages/`.

### Measurements behind the rules (this workstation, 20 sheets)

`tools/bench_layouts.py` times OCR and parsing for layouts of COPIESxSLOTS:

| Layout (one card unless noted) | Qwen3-VL 2B images/min | Qwen3-VL 8B images/min |
|---|---|---|
| 1 engine x 1 slot | 5.1 | 2.9 |
| 1 engine x 2 slots | 7.6 | 4.3 |
| 1 engine x 4 slots | 9.4 | 5.7 |
| 1 engine x 8 slots | 9.0 | 5.4 |
| 4 engines x 1 slot, same card | 5.3 | 3.0 |
| 2 engines x 4 slots, same card | 9.2 | 5.4 |
| 2 engines x 4 slots, one per card | not run | 9.8 |

Label detection (section 15) takes about 0.3 s per image with Core ML on an M2 Pro and
about 1.3 s with onnxruntime on its CPU. On Linux with onnxruntime it takes about 0.35 s on
a 32-core Threadripper PRO 5975WX and about 0.8 s when limited to 8 threads. It is not a
bottleneck next to OCR.

## 12. Download queue and split model files

### Queue

Model downloads run one after another. `POST /v1/models/local/<build_id>/download`
while another download runs no longer returns 409: the new job is created in
state `queued` and starts by itself when the ones ahead of it end.

- Job objects gain `queue_position` (1 = next to run; `null` unless queued).
- `install_state` gains `queued`; `unavailable_reason` is "This model is
  waiting to download."
- 409 `CONFLICT` only when the same build is already downloading or queued.
- The disk-space check counts the remaining bytes of every download ahead in
  line; the 507 details add `queued_bytes`.
- `POST /v1/models/local/<build_id>/download/cancel` on a queued download takes
  it out of line at once (state `canceled`).
- Deleting a build that is downloading or queued is 409.
- Engine installs still allow one at a time (409 for a second).
- Queues live in memory: after a restart, partial downloads show as `partial`
  and are resumed from the Models tab.
- While a file downloads, `<file>.partial.lock` holds an exclusive lock, so two
  VoucherVisionLocal processes sharing a models folder cannot write the same
  file; the second is refused with `DOWNLOAD_FAILED`.

### Split model files

A build's `files` may list the numbered parts of one split model
(`x-00001-of-00003.gguf`, ...) instead of one `model` file. Parts must be
complete, in order, and in one folder; file paths may include a subfolder
(`BF16/x-00001-of-00002.gguf`). The engine is given the first part.

### Thinking models

Families whose models think before answering (Qwen3.5/3.6, Gemma 4, MiMo, Muse)
start the engine with `--reasoning off` (registry `inference.server_args`), so
OCR and parsing answer directly.

## 13. Per-step model settings (`/v1/generation`)

The OCR and Parsing cards on the VVL tab each have one row of setting chips
under their model list (`ui/js/tabs/gen_settings.js`). Data:
`vvl/registry/model_settings.json`; logic: `vvl/generation.py`. Settings key
`generation`:

```yaml
generation:
  ocr:   {reasoning: off|on, strength: low|medium|high|xhigh,
          output_length: standard|long, image_detail: standard|high}
  parse: {reasoning: off|on, strength: low|medium|high|xhigh, output_length: standard|long}
  context: {"local:<build>": 32768}     # absent = automatic
```

`GET /v1/generation` → for the saved selection (local models only):

```json
{"ocr":   {"models": [{"model_id": "...", "label": "...",
                       "controls": {"reasoning": {"options": ["off","on"], "model_default": "on",
                                                  "can_turn_off": true,
                                                  "literal": {"off": {...}, "on": {...}}, "note": "..."},
                                    "strength": {"options": ["low","medium","high","xhigh"], "default": "high"},
                                    "output_length": {"options": [...], "max_tokens": {"standard": 3072, "long": 8192}},
                                    "image_detail": {"options": [...], "tokens": {...}}},
                       "values": {...}, "sent": {"max_tokens": 3072, "temperature": 0.7, "...": "..."}}],
           "values": {"reasoning": "off", "strength": "high", "output_length": "standard", "image_detail": "standard"},
           "applicable": {"reasoning": false, "strength": false, "output_length": true, "image_detail": true}},
 "parse": {"...": "..."},
 "context": {"min": 8192, "models": [{"model_id": "...", "label": "...", "steps": ["ocr","parse"],
                                      "min": 8192, "max": 262144, "value": "auto" | 32768,
                                      "mode": "auto" | "auto_max" | "fixed", "needed": 8192, "effective": 8192}]}}
```

- A control appears only for models that support it. `sent` is exactly what a
  request sends (plus image token limits for OCR, which are engine options).
- Reasoning and strength are sent per request; changing them never restarts
  the engine.
- `strength` appears only for families whose reasoning lists `levels` (Muse).
  While reasoning is on, the chosen level replaces `{strength}` in
  `literal.on`; with reasoning off the family's `literal.off` is sent (Muse:
  `low`). A family with `can_turn_off: false` (Muse) still reasons with
  reasoning off, so its `max_tokens` are the reasoning-on limits.
  Image detail and the context window are engine options: a change restarts
  that model's engine at its next request.
- `PUT /v1/generation` takes any of `ocr`, `parse` (partial objects, merged)
  and `context` (`{"local:<build>": tokens | "auto"}`, merged; `"auto"`
  removes the entry). Returns the same shape as GET. Invalid values are 400.
- Sampling (temperature, top_p, ...) is not a user setting: each request sends
  the model card's defaults from `model_settings.json`.
- The engine-wide `engine.context_size` minimum is now 8,192.

## 14. OpenRouter (`api:openrouter:<model>`)

OpenRouter replaces the planned Hyperbolic provider: one key reaches the hosted
models of every major family (Claude, GPT, Gemini, Mistral, Qwen, and several
hundred more). It is the first API provider that can run. Code:
`vvl/openrouter.py` (catalog, cost estimate, request limiter),
`vvl/engines/openrouter.py` (engine), `vvl/api/openrouter.py` (routes).

**How a request runs.** Exactly like a local model, except that the two model
calls go to `https://openrouter.ai/api/v1/chat/completions` instead of
llama.cpp, through the same OpenAI-compatible engine. OCR sends the image as a
base64 data URL. Parsing sends the prompt's JSON schema as `response_format`
(with `provider.require_parameters`) when the model lists `structured_outputs`;
if OpenRouter has no endpoint that takes the schema (HTTP 400, 404, or 422) the
request is sent once more without it. The thinking level chosen on the VVL tab
(`ocr_thinking_level`, `llm_thinking_level`: low, medium, high) is sent as
`reasoning: {"effort": <level>, "exclude": true}` for models that list
`reasoning`. Output is limited to 8,192 tokens (16,384 for models that think),
or the model's own limit when that is lower. Cost fields are tokens times the
catalog price; reasoning tokens are reported as `thinking_cost`.

**Model list.** `GET https://openrouter.ai/api/v1/models` is public, so the
list needs no key. It is cached in `<VVL_HOME>/cache/openrouter_models.json`
and refreshed once a day or on request. A model is kept when it takes text and
returns only text, and has a fixed price. Roles:

| Role | Needs |
|---|---|
| OCR | text and image input |
| Parsing | text input (every kept model) |

The pickers on the VVL tab list only the models the user added
(`openrouter.models` in settings); a model that cannot read images is offered
for parsing only, and `PUT /v1/selection` refuses it for OCR.

**Estimated cost per 1,000 images** uses the token counts of the
VoucherVisionGO cost calculator and the model's OpenRouter price:

| Step | Typical (tokens in / out) | Long labels (in / out) |
|---|---|---|
| OCR | 1,300 / 150 | 1,500 / 1,000 |
| Parsing | 2,350 / 650 | 3,000 / 1,000 |

Thinking tokens are extra and are billed at the output price.

**Private providers.** OpenRouter's public zero-data-retention list (`GET /endpoints/zdr`, the green shield on
openrouter.ai) marks each model `private` (true when at least one provider keeps no copy of the request) and
lists those `private_providers`. With `openrouter.private_only` on (`PUT /v1/settings`), the gallery shows only
private models and every request carries `provider: {"zdr": true}`, so OpenRouter routes only to such a
provider; a model without one fails with a plain message. Verified 2026-11-20 with one real OCR request to
`anthropic/claude-opus-5.5` pinned to Amazon Bedrock (4.7 s; the reported cost matched OpenRouter's bill).

**Models tab.** Two half-width tabs in one container: Local models (green) and OpenRouter models (amber). The
OpenRouter tab is the gallery (`ui/js/tabs/openrouter_gallery.js`): a filter rail (use, search, owner, private
providers, cost, release date, abilities) and a sortable table.

**Concurrent requests.** `openrouter.concurrency` (1 to 4, default 4) is the
most OpenRouter requests this app runs at one time. It is enforced in the
server (`vvl.openrouter.limiter`), so a batch that sends more simply waits its
turn, and the batch slider on the VVL tab follows it. OCR and parsing requests
share the limit. HTTP 429 and 5xx answers are retried up to 4 times with
backoff (2, 5, 10, 20 s), honoring `Retry-After`. OpenRouter does not set a
fixed requests-per-minute limit for paid models; free models (`:free`) are
limited to about 20 requests a minute. Errors: 401 -> `UPSTREAM_AUTH_FAILED`,
402 and 429 -> `UPSTREAM_QUOTA_EXHAUSTED`.

### Routes

`GET /v1/openrouter/catalog[?refresh=1]`

```json
{
  "models": [{
    "id": "google/gemini-3.8-flash", "model_id": "api:openrouter:google/gemini-3.8-flash",
    "name": "Gemini 3.8 Flash", "full_name": "Google: Gemini 3.8 Flash",
    "owner": "google", "owner_label": "Google", "created": 1790000000,
    "context_length": 1048576, "max_output_tokens": 65536,
    "input": ["text", "image"], "vision": true, "roles": ["ocr", "parse"],
    "price_in": 0.825, "price_out": 4.125, "free": false,
    "reasoning": true, "structured_output": true, "json_mode": true, "expires": null,
    "description": "...",
    "cost_1k": {"ocr": {"typical": 1.6913, "high": 5.3625}, "parse": {"typical": 4.62, "high": 6.6}}
  }],
  "owners": [{"id": "openai", "label": "OpenAI", "count": 106, "vision": 97}],
  "fetched_at": 1795190000.0, "stale": false, "error": null,
  "added": ["google/gemini-3.8-flash"], "concurrency": 4, "concurrency_range": [1, 4],
  "configured": true, "selection": {"ocr": ["..."], "parse": "..."},
  "cost_tokens": {"ocr": {"typical": {"in": 1300, "out": 150}, "high": {"in": 1500, "out": 1000}}, "parse": {"...": "..."}},
  "key_url": "https://openrouter.ai/settings/keys"
}
```

Prices are USD per 1,000,000 tokens; `cost_1k` is USD per 1,000 images
(`ocr` is null for a model that cannot read images). When a refresh fails, the
previous list is returned with `error` set.

`PUT /v1/openrouter/models` with `{"models": [...]}` (the whole list) or
`{"add": [...], "remove": [...]}`. Names must be in the catalog. Removing a
model that is chosen for OCR or parsing resets that choice. Answers with the
catalog payload.

`GET /v1/openrouter/account` -> `{"configured", "ok", "usage_usd", "limit_usd",
"limit_remaining_usd", "is_free_tier", "message"}` from OpenRouter's `GET /key`.
The key itself is never returned.

`PUT /v1/settings` accepts `{"openrouter": {"concurrency": 1..4}}`.

`GET /v1/models`: OpenRouter's entry of `api.providers` has
`models_state: "catalog"`, the added models (with `roles` and `capabilities`),
`concurrency`, and `runnable` (true once a key is saved). The key test is
`GET {base_url}/key`.

Gemini, Anthropic, and OpenAI stay listed as providers that cannot run yet;
their models are reachable through OpenRouter.

## 15. Label detector (`/v1/detector`)

TextCollage's boxes come from LeafMachine3's archival detector (YOLO26x at 1280,
end2end, nine classes: Ruler, Barcode, Colorcard, Label, Map, Envelope, Photo,
Attached Item, Weights; `Label`, `Barcode` and `Map` go into the collage). It is
pinned in `vvl/registry/archival_detector.lock.yaml` and not bundled:

- `vvl serve` starts its download at first start (job kind `detector_download`,
  label "Label detector"; `VVL_DETECTOR_AUTO_DOWNLOAD=0` turns that off). Files go
  to `<VVL_HOME>/detector/<repo>@<revision>/`; each is checked by sha256, then
  `verified.json` records them. After the job, the processor loads the detector.
- Format: `coreml_fp32` on macOS (Core ML, Apple GPU, 223 MB), `onnx` elsewhere
  (onnxruntime, CPU, 224 MB). `VVL_DETECTOR_FORMAT` overrides it.
- NVIDIA GPU (Linux x86_64 and Windows x64): when the system report shows a usable
  NVIDIA GPU, `vvl serve` downloads the detector's GPU runtime after the detector
  (job kind `detector_gpu_download`, label "GPU runtime"): onnxruntime-gpu 1.26.0
  (CUDA 12) and the NVIDIA wheels it loads (cuBLAS, cuDNN 9, ...), 2.0 GB on Linux
  and 1.9 GB on Windows, 3.1 GB installed. They are pinned in
  `vvl/registry/detector_gpu.lock.yaml` (generated from `uv.lock` by
  `tools/lock_detector_gpu.py`), checked by sha256 and installed with uv into
  `<VVL_HOME>/runtime/detector-gpu/`, never into the app's environment (the app's
  onnxruntime and onnxruntime-gpu are the same Python package). The `onnx` model then
  runs in a worker process (`textcollage/cuda_worker.py`) on the GPU with the most
  free memory: about 0.06 s per sheet instead of 0.4 s, the same boxes as the CPU
  (within 0.2 px), about 1.9 GB of GPU memory. The switch happens without a restart.
  Usable means: NVIDIA driver 525.60.13 or newer (Windows 527.41), compute
  capability 6.0 or newer, and on Linux glibc 2.27 or newer. Only the driver comes
  from the system. When the GPU fails (at start or mid-batch), the detector goes
  back to the CPU and says why in `gpu_error`. `VVL_DETECTOR_GPU=0` turns the GPU
  runtime off.
- Until it is installed (or when it fails to load), `/process` uses the whole
  image: `success.text_collage` is `"False"` and `position_original` is `{}`.
  No request fails because of the detector.

`GET /v1/detector` →

```json
{"state": "installed", "format": "coreml_fp32", "runtime": "Core ML, Apple GPU",
 "repo": "phyloforfun/lm3_archival_detector__yolo26x_det_1280", "revision": "cf178fe…",
 "size_bytes": 222904317, "path": "…/detector/lm3_archival_detector__yolo26x_det_1280@cf178fe1d447",
 "job": null, "error": null,
 "loaded": true, "loaded_runtime": "Core ML, Apple GPU", "gpu_error": null, "load_error": null,
 "gpu": {"state": "not_needed", "reason": "not_supported", "message": "There is no NVIDIA runtime …", …}}
```

- `state`: `not_installed | downloading | installed | error` (`error` = the last
  download failed; `error` is its `{code, message}`).
- `loaded_runtime`: what the loaded detector runs on, e.g. `"ONNX Runtime, CPU"` or
  `"ONNX Runtime, NVIDIA GeForce RTX 4060 Laptop GPU"`. `gpu_error`: why the GPU is
  not used although its runtime is installed (else null).
- `gpu`: the NVIDIA runtime. `state`: `not_needed | disabled | not_installed |
  downloading | installed | error`. `not_needed` carries `reason` (`not_supported`,
  `no_nvidia_gpu`, `driver_too_old`, `gpu_too_old`, `glibc_too_old`) and `message`.
  Otherwise also `device` (`{index, name, vram_free_bytes, vram_total_bytes}`, the
  nvidia-smi index), `size_bytes`, `onnxruntime_gpu`, `cuda`, `path`, `job`, `error`.

`POST /v1/detector/install` → 202 `{"job": Job}`: starts (or retries) the
download, or returns the job already running. 409 `CONFLICT` when installed.

`POST /v1/detector/gpu/install` → 202 `{"job": Job}`: the same for the GPU runtime.
409 `CONFLICT` when it is installed, or when `gpu.state` is `not_needed` or `disabled`.

`vvl detector [--download] [--format coreml_fp32|onnx]` shows the same state from
the command line and can download the files in the foreground.

## 16. Saved results

Every run can be saved to disk. The desktop app does this for every press of Run.

**The folder.** `output_dir` in the settings (section on `/v1/settings`; an absolute
path, `null` = the default `<VVL_HOME>/runs`).

- `GET /v1/output` → `{dir, default_dir, is_default, exists, synced_folder, subfolders, stamp_format}`.

**One folder per run.**

- `POST /v1/output/batches` with `{"name": "<folder name or null>", "append": false}` → 201
  `{id, path, folder, name, append, existed, saved, xlsx, errors}`.
  - `append: false` (default): a new folder, `YYYYMMDD-HH-MM-SS`, or
    `<name>_YYYYMMDD-HH-MM-SS` with a name. It never reuses an existing folder (a second
    run in the same second gets `-2`).
  - `append: true`: exactly `<name>` (required). It is created if missing; files with
    the same name are replaced, and the spreadsheet keeps the rows already saved there.
  - Names may not contain `< > : " / \ | ? *`, start with a dot, end with a dot or
    space, be a Windows reserved name, or exceed 100 characters: 400 `INVALID_INPUT`.
  - A folder that cannot be created or written: 400 `OUTPUT_DIR_UNAVAILABLE`.
- Send the `id` as `output_batch` with `/process`, `/process-url` and
  `/v1/process/finish`. Without it nothing is saved. An unknown id: 404
  `OUTPUT_BATCH_NOT_FOUND`, before anything runs.
- `POST /v1/output/batches/<id>/finish` → the same object, after the spreadsheet is
  written a last time. `saved` counts the files per subfolder (`xlsx` counts rows);
  `errors` lists files that could not be saved. Saving never fails a request.

**What a run's folder holds** (file names follow the image's name; a second image with
the same name in one run gets `_2`):

| Subfolder | File | Content |
|---|---|---|
| `json/` | `<image>.json` | the full result, exactly as the API returned it |
| `xlsx/` | `results.xlsx` | one row per parsed record |
| `label_collage/` | `<image>.jpg` | the label collage sent to OCR (`collage_info.base64image_text_collage`) |
| `input_image/` | `<image>.<ext>` | the image that was processed |
| `ocr_txt/` | `<image>.txt` | the result's `ocr` text |

In a two-pass batch the images are saved in the OCR pass and the rest in the parsing pass.

**The spreadsheet** follows VoucherVisionGO's `save_results_to_xlsx`: a sheet named
`results`, `filename` (no extension) first, then the record's fields in the order they
first appear, every cell stored as text so Excel converts nothing. Each cell passes
VoucherVisionGO's write-time guards (`vvl/vendor_vv/OCR_sanitize.py`): lists and objects
are flattened, XML-illegal characters and line breaks removed, a leading `=`, `+` or `@`
is neutralized with `'`, and text is cut at Excel's 32,767 character limit. OCR-only and
notebook runs have no record and add no row. If `results.xlsx` cannot be replaced (open
in Excel on Windows), the records are written to `results_<timestamp>.xlsx` and reported
in `errors`. No CSV is written: VoucherVisionGO marks its CSV export "use at your own
risk" because Excel converts its cells.
