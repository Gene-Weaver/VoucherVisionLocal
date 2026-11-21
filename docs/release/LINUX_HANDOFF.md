# Linux handoff: after the Mac release work (2026-11-20)

You are continuing VoucherVisionLocal (VVL) on Linux after a Mac session.
The Mac work for the first release is done (results: `docs/release/MAC_HANDOFF.md`,
section "Done on the Mac"). `main` now also carries the 0.2.0 label detector,
which was built and checked on the Mac and now needs Linux. Release 0.2.0 only
after the checks below pass; releases are made on the Mac.

## Rules (from the owner, unchanged)

- Never open, read, print or grep `PRIVATE.yaml` or any `.env.signing`.
- Commits go directly on `main`, small, with no `Co-Authored-By` or other
  attribution line. American English. Push when done.
- Never download a local LLM model without asking the owner first. The label
  detector below (224 MB) is part of the app and is fine to download.
- Test runs that start the app or the server use scratch `VVL_HOME` (and
  `VVL_USER_DATA_DIR` for the desktop app), never the owner's real data folder.
- No personal names, emails or home-folder paths in repo files (the repo is
  public); the `author` field in `desktop/package.json` and LICENSE are the only
  exceptions.

## 1. Get in sync

The Mac worked in its own clone outside Dropbox and pushed to GitHub. It did
not touch the Dropbox copy, so get the new commits from GitHub:

```bash
git status                       # commit (or stash) your own work first
git pull --rebase origin main    # from ea536da to the current main
uv sync                          # uv.lock changed: openvino removed, coremltools (macOS only) added
```

Run git for this repo on one machine at a time: `.git` is in Dropbox, so two
machines writing it at once can corrupt it.

## 2. What is new on `main`

| Commit(s) | What |
|---|---|
| Release for Apple Silicon, Windows and Linux | `deploy.sh` builds the Windows NSIS installer (unsigned) and the Mac for Apple Silicon only; checks every updater feed; uploads blockmaps. See `docs/RELEASING.md`. |
| DMG window | Background with an arrow (`desktop/scripts/make_dmg_background.py`, Mac only). |
| Versions 0.1.1, 0.1.2 | Updater test releases. `VERSION` is `0.1.2`. |
| TextCollage: LeafMachine3's archival detector | The 0.2.0 work, below. |
| Setup page and job bar; docs | The detector download in the UI; `docs/API.md` section 15. |
| Privacy fix | The Mac handoff says "the owner"; a vendored demo no longer has local paths. |
| Main window opens maximized | `desktop/main.js`: maximized, not full screen (un-maximize gives 1280x860); screenshot runs keep 1280x860. Check it on Linux and Windows. |

Published on GitHub: v0.1.0, v0.1.1, v0.1.2. Each has the Apple Silicon
DMG and zip (signed, notarized), `VoucherVisionLocal-Setup-<v>-x64.exe`
(unsigned, never run on Windows: the owner will test it) and
`VoucherVisionLocal-<v>-x86_64.AppImage` (built on the Mac, **never started on
Linux yet**), plus `latest-mac.yml`, `latest.yml`, `latest-linux.yml`.

## 3. The label detector (0.2.0)

TextCollage's boxes now come from LeafMachine3's archival detector,
`phyloforfun/lm3_archival_detector__yolo26x_det_1280` (YOLO26x at 1280, end2end,
no NMS). TextCollage's own YOLOv12n and OpenVINO are gone. Still no torch.

- `textcollage/detector.py`: letterbox, box scaling and the end2end tail,
  ported from LM3's `ultra_replacements.py` / `coreml_native.py`. Two runtimes:
  `onnx` (onnxruntime, CPU; Linux and Windows) and `coremltools` (macOS, Apple GPU).
  `textcollage/CollageEngine.py` keeps its collage code; only detection changed.
- `vvl/label_detector.py` + `vvl/registry/archival_detector.lock.yaml`: the pin
  (Hugging Face revision `cf178fe` + sha256 of every file) and a
  `detector_download` job. `vvl serve` starts it at first start
  (`VVL_DETECTOR_AUTO_DOWNLOAD=0` turns that off); files go to
  `<VVL_HOME>/detector/`. Until it is installed, `/process` uses the whole image
  (no more 503). `VVL_DETECTOR_FORMAT=onnx|coreml_fp32` overrides the format.
- API: `GET /v1/detector`, `POST /v1/detector/install` (API.md section 15).
  CLI: `vvl detector [--download] [--format onnx|coreml_fp32]`.
- Tests: the app fixtures use a fixed-box fake detector
  (`tests/fakes/fake_detector.py`); `tests/test_label_detector.py` covers the
  math, the download job, the routes and, with `VVL_TEST_DETECTOR_HOME`, the
  real model.

Measured on the Mac (M2 Pro): every detection on 21 sheets matches LM3's own
predictor (279 detections, boxes within 0.001 px); about 0.33 s per sheet with
Core ML and 1.3 s with onnxruntime on the CPU. The packaged app downloaded the
detector by itself in 14 s at first start and then built a real collage.

## 4. Your tasks

1. **Detector on Linux.**
   ```bash
   export VVL_HOME=/tmp/vvl-detector-home          # any scratch folder
   uv run vvl detector --download                  # onnx, 224 MB, sha256-checked
   VVL_TEST_DETECTOR_HOME=$VVL_HOME uv run pytest tests/test_label_detector.py
   ```
   `test_real_onnx_matches_the_mac_reference` compares the detections on the
   fixture image with the Mac's (same classes and order, scores within 0.005,
   boxes within 0.5 px). If it fails, report the numbers; do not loosen it
   without the owner. Note the time per image on this CPU for API.md.
2. **Full test suite** (`uv run pytest`). On the Mac: 709 passed, 2 failed, 3
   skipped. The two failures assume a Linux host and should be fixed to pass on
   macOS too: `test_engine_routes.py::test_doctor_reports_engine_and_models`
   (expects `linux-x64-cuda12`) and
   `test_installer.py::test_install_route_and_disk_check` (installs
   `linux-x64-cpu`, which a Mac cannot pick: 409 instead of 507). Pin the host
   platform in those tests the way `conftest.py` pins glibc.
3. **The released AppImage** (handoff step 6, never done on Linux): download
   `VoucherVisionLocal-0.1.0-x86_64.AppImage` from the v0.1.0 release, run it
   with scratch `VVL_HOME`/`VVL_USER_DATA_DIR`, check the first-launch Python
   setup and a demo run, then check that it offers v0.1.2 ("Update ready") and
   restarts as 0.1.2. Ubuntu 24.04+ needs `libfuse2t64`.
4. **An AppImage of current `main`** (`cd desktop && npm ci && npm run dist`),
   first start with an empty `VVL_HOME`: the detector downloads by itself (job
   bar: "Downloading the label detector"), `GET /v1/detector` reaches
   `installed` with `loaded: true`, and a demo run returns
   `success.text_collage == "True"` with boxes in `collage_info.position_original`.
5. **Small 0.2.0 polish, if the owner agrees:**
   - The Setup page shows the detector step only after a "How will you run
     models?" choice, so a fresh install sees the download only in the header.
     Show the step (or a line) before the choice.
   - The header job pill cuts "Downloading the label detector · 40%" short;
     a shorter label would fit.
6. **Report back**: commit on `main` (no attribution trailer), push, and say what
   the owner needs to do. The Mac then bumps `VERSION` to `0.2.0` and runs
   `./deploy.sh`. The update 0.1.2 -> 0.2.0 changes `uv.lock`, so the first start
   after it re-runs the Python setup and then downloads the detector: worth
   checking on Linux too once 0.2.0 is out.

## Facts worth knowing

- The Mac builds releases in its own clone outside Dropbox; do not expect Mac
  build outputs in the Dropbox copy.
- From a VS Code terminal, `ELECTRON_RUN_AS_NODE=1` is set; Electron then acts
  as plain Node. `desktop/launch.js` removes it for `npm start`; run a built app
  with `env -u ELECTRON_RUN_AS_NODE ...`.
- Intel Macs are not built: `pymupdf` 2.1 and `onnxruntime` 1.32.0 have no macOS
  x86_64 wheels (RELEASING.md says how to change that). `coremltools` has them.
- Windows is unsigned like the Editor's; `WIN_CSC_LINK` / `WIN_CSC_KEY_PASSWORD`
  in `.env.signing` would sign it. The detector on Windows is the `onnx` format;
  `textcollage.detector.imread` reads non-ASCII paths there.
- The VoucherVisionGO-Editor's releases list only the x64 zip in
  `latest-mac.yml` (reported to the owner; do not fix the Editor unless asked).
