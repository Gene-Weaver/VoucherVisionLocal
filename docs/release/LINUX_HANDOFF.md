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

## Done on Linux (2026-11-20)

Checked on Ubuntu 22.04, x86_64 (Threadripper PRO 5975WX, libfuse2 installed).

- **Detector.** `vvl detector --download` fetched and verified the onnx format in
  4 s; `test_real_onnx_matches_the_mac_reference` passes, so Linux and the Mac give
  the same detections. Time per sheet with onnxruntime: 0.35 s (all threads),
  0.78 s (8 threads); in `docs/API.md`.
- **OpenVINO, measured again.** On 21 test sheets of the archival dataset, OpenVINO
  2026.4.1 running the same `model.onnx` gave identical detections (0 px, 0 score
  difference) and was 18% faster with all threads (0.29 s vs 0.35 s) and 8% faster
  with 8 threads (0.72 s vs 0.78 s). The OpenVINO IR in the Hugging Face repo is
  fixed at 1280x1280, so it needs square padding: slower (0.39 s) and different
  boxes on 19 of 21 sheets. onnxruntime stays installed anyway (`pymupdf` 2.1
  needs it), so OpenVINO would be a second runtime and a 30-85 MB larger first
  setup. Kept onnxruntime; the owner may decide otherwise.
- **Tests.** 712 passed, 3 skipped. The two tests that failed on the Mac now use the
  Linux workstation system report, so they pass on any host (checked with a fake
  macOS report: both failed before, pass after).
- **Released AppImage.** v0.1.0 from the release starts, runs the first-launch
  Python setup, and a demo run returns a collage. It found v0.1.2 at launch
  (differential download, 129 KB of 152 MB), showed "Update ready", and the click
  replaced the file (sha512 equals the v0.1.2 feed) and restarted as v0.1.2. Closing
  the window leaves no backend process.
- **AppImage of current `main`.** First start with an empty data folder downloads
  the detector by itself (about 3 s); `GET /v1/detector` is `installed`, the demo
  run returns `text_collage: "True"` with barcode and label boxes in
  `collage_info.position_original`. The window opens maximized (1850x1136 on an
  1850x1173 work area).
- **0.2.0 polish (done).** The Setup page lists the app, Python and the label
  detector before the "How will you run models?" choice, followed by "Choose an
  option above to see the rest of what is needed." The header pill now says
  "Label detector · 40%" and "Engine install · 40%" (both fit the 190 px narrow
  header; the tooltip has the long form).
- **Dev tools.** `tools/dev_fake_backend.py` serves a fake 8 MB label detector and
  uses the fixed-box stand-in, so `scenarios/setup-page.js` reaches the "ready"
  banners again (it had 2 failures since 0.2.0); 48 screenshots, 0 problems.

## Next on the Mac

Bump `VERSION` to `0.2.0`, commit, push, `./deploy.sh`. Then on both machines:
start 0.1.2, take the update, and check that the first start of 0.2.0 re-runs the
Python setup (`uv.lock` changed) and then downloads the label detector.

## Done on the Mac (0.2.0, 2026-11-21)

- **Released v0.2.0** (`cf16814`): Apple Silicon DMG and zip (signed, notarized,
  stapled), the unsigned Windows installer and the AppImage, with all three updater
  feeds pointing at 0.2.0. The release notes start with a short "What's new".
- **Tests on the Mac:** 711 passed, 4 skipped, 0 failed (the two engine tests pass
  now); the real-model detector tests pass in both formats.
- **Update 0.1.0 -> 0.2.0** on a scratch copy of the released 0.1.0: differential
  download (3.6 MB of 131 MB), installed on quit. The first start of 0.2.0 re-ran
  the Python setup (new `uv.lock`), downloaded the detector by itself in 12 s
  (Core ML, Apple GPU), and a demo run returned `text_collage: "True"` with 7
  label and 1 barcode boxes. The owner's installed 0.1.2 took the update the
  normal way. The Setup page and the "Label detector · 21%" pill look right.
- **Gotcha for update tests on macOS:** quit by closing the window, not with
  SIGTERM. A SIGTERM in the first ~15 s after "Update ready" (while Squirrel is
  still staging the update) held the quit until Squirrel finished, and ShipIt then
  relaunched the app through LaunchServices, without the test's `VVL_*` settings.
  A window close at that moment quits at once, and the update installs at a later
  quit.

**Next on Linux:** start the v0.1.2 AppImage with scratch folders, take the update
to 0.2.0, and check that its first start re-runs the Python setup and then
downloads the label detector. Kept onnxruntime (not OpenVINO) on Linux and
Windows, as measured above; the owner may still choose OpenVINO.

## Done on Linux (0.2.0, 2026-11-21)

Update 0.1.2 -> 0.2.0 on the AppImage that had updated itself from the released
0.1.0 (its Python environment set up by 0.1.x), with scratch folders:

- 0.1.2 found 0.2.0 at launch: differential download, 583 KB of 144 MB. Closing the
  window installed it without a relaunch; the file's sha512 equals the v0.2.0
  `latest-linux.yml`.
- First start of 0.2.0: "Updating VoucherVisionLocal: This version uses updated
  Python packages" (new key in `runtime/runtime.json`; `openvino` is gone from the
  environment), then the label detector downloaded by itself in about 5 s ("Label
  detector · 19%" ... "85%" in the header) and loaded.
- Demo run: `text_collage: "True"`, 7 label and 1 barcode boxes. Closing the window
  leaves no backend process.

Nothing left for Linux from this handoff.
