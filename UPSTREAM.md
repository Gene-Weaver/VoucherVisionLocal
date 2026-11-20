# Where the copied code came from

VoucherVisionLocal started from working code in sibling repositories. Files were
copied once on 2026-11-19 and are expected to diverge; there is no sync tooling.
This file only records provenance.

| Here | Copied from | Source commit |
|---|---|---|
| `app.py` | VoucherVisionGO `app.py`, cut down to the local server (11,640 lines to about 500); the borrowed pipeline, image, fetch, error-envelope, and prompt-listing code moved into `vvl/pipeline.py`, `vvl/images.py`, `vvl/fetch/policy.py`, `vvl/errors.py`, and `vvl/prompts.py` | `2658891` |
| `vvl/fetch/anti_bot_fetch.py`, `vvl/fetch/guarded_fetch.py`, `vvl/fetch/url_name_parser.py` | VoucherVisionGO `anti_bot_fetch.py`, `guarded_fetch.py`, `url_name_parser.py` (only change: `anti_bot_fetch` imports `vvl.fetch.url_name_parser`) | `2658891` |
| `vvl/enrichment/wfo_local.py` | VoucherVisionGO `wfo_local_lookup.py` (unchanged; needs the optional `wfo` extra, imported only on demand) | `2658891` |
| `vvl/vendor_vv/utils_LLM_JSON_validation.py` | VoucherVision `vouchervision/utils_LLM_JSON_validation.py` (unchanged) | `78753a4` |
| `vvl/engines/mock.py` (`SAMPLE_OCR_TEXT`, `SAMPLE_RECORD`) | values from `tests/fixtures/vvgo_results_sample.json` (VoucherVisionGO `results.json`) | `2658891` |
| `tests/fixtures/vvgo_results_sample.json` | VoucherVisionGO `results.json` | `2658891` |
| `tests/fixtures/images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg` | VoucherVisionGO `demo/images/` | `2658891` |
| `docs/VoucherVision.reference.yaml` | VoucherVisionGO `VoucherVision.yaml` | `2658891` |
| `vvl/vendor_vv/*.py` | VoucherVision `vouchervision/` (`prompt_catalog.py`, `OCR_sanitize.py`, `OCR_resize_for_VLMs.py`, `OCR_Prompt_Catalog.py`, `general_utils_slim.py`, `vouchervision_main_slim.py`) | `78753a4` |
| `vvl/registry/api_cost.yaml` | VoucherVision `api_cost/api_cost.yaml` | `78753a4` |
| `prompts/builtin/*.yaml` (18 files) | VoucherVision `custom_prompts/` (git-tracked files only) | `78753a4` |
| `textcollage/CollageEngine.py`, `textcollage/models/openvino/*` | TextCollage (working tree, based on `4b697e8`) | `4b697e8` |
| `ui/` (page structure, styles, scripts) | VoucherVisionGO `webpage/` | `2658891` |

Local edits to copied files: `vvl/vendor_vv/prompt_catalog.py` opens prompt
files as UTF-8 explicitly (Windows default encodings broke non-ASCII prompts),
and `textcollage/CollageEngine.py` blocks OpenVINO's import-time Google
Analytics telemetry for this process before importing `openvino`.

The TextCollage detector weights (`textcollage/models/openvino/`) are an
Ultralytics YOLO export and carry the AGPL-3.0 license noted in their
`metadata.yaml`.

Add a row here whenever another file is copied in.
