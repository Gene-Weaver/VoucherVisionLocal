# Mac handoff: sign, notarize, and make the first VoucherVisionLocal release

You are finishing the release setup for VoucherVisionLocal (VVL) on a Mac.
Everything that does not need macOS was built and tested on Linux first. Read
`docs/RELEASING.md` before starting: it describes how the installer, the
first-launch Python setup and the updater work.

## Rules (from the owner)

- `PRIVATE.yaml` holds the owner's real API keys. Never open, read, print, grep or
  otherwise view it. Avoid repo-wide `grep -r` / `cat *` that could print it.
- `.env.signing` (here and in `~/Dropbox/VoucherVisionGO-Editor`) holds signing
  secrets. Never print it, paste it, or commit it. Copy values from the Editor's
  file into this repo's `.env.signing` without displaying them (for example
  `cp ~/Dropbox/VoucherVisionGO-Editor/.env.signing .env.signing`, after
  checking with `grep -c '^export APPLE_' ` that it has the APPLE_* names).
- Commits go directly on `main`. Never add a `Co-Authored-By` or any Claude
  attribution line to commit messages. American English everywhere.
- Never download a local AI model without asking the owner first (they are tens of GB).
- Mac test runs that start the app must use scratch data: set `VVL_HOME` and
  `VVL_USER_DATA_DIR` to scratch folders, unless you are testing the real
  first-run experience in a fresh macOS user account.

## Already done (Linux, 2026-11-20)

- `desktop/package.json` `build` block modeled on the Editor (appId
  `com.vouchervision.local`, product name `VoucherVisionLocal`, dmg + zip for
  arm64 + x64, hardened runtime, `build-resources/entitlements.mac.plist`
  copied from the Editor, `notarize: true`, GitHub publish).
- `deploy.sh` modeled on the Editor's, with these changes: both Mac
  architectures build in ONE electron-builder run (the Editor's parallel
  per-arch runs leave `latest-mac.yml` listing only the x64 zip); a preflight
  (clean tree, `main` == `origin/main`); a signed-and-Gatekeeper check before
  release; the release is created at the exact commit (`--target`).
- Linux AppImage built, first-launch Python setup tested, the demo pipeline run
  from the read-only bundle, and both update paths tested with a local feed.
- Not tested anywhere yet: anything on macOS.

## Your tasks

1. **Prerequisites.** Xcode command line tools, Node 22+, `gh auth status` OK
   for Gene-Weaver, the "Developer ID Application" certificate in the login
   keychain (the same one the Editor uses; check with
   `security find-identity -v -p codesigning`). `git pull` on `main`.
2. **Signing secrets.** Create `.env.signing` from `.env.signing.example`
   (see the rule above for copying from the Editor). Confirm
   `git check-ignore .env.signing` prints the path.
3. **Unsigned smoke test** (catches problems before Apple is involved):
   `cd desktop && npm ci && npm run dist -- --mac --arm64 -c.mac.identity=null -c.mac.notarize=false`,
   then run `build/mac-arm64/VoucherVisionLocal.app` with scratch
   `VVL_HOME`/`VVL_USER_DATA_DIR`. Check: the "Setting up VoucherVisionLocal"
   screen, Python installs, the app opens on the Setup page, a demo run on the
   VVL tab works (demo engine), Quit stops the backend (no leftover `python`
   process).
4. **Signed build without releasing:** `./deploy.sh --skip-release`. Check:
   - both `.app` builds pass `codesign --verify --deep --strict --verbose=2`
     and `spctl --assess --type execute -vv`;
   - the bundled uv is signed with the hardened runtime:
     `codesign -dv --verbose=4 desktop/build/mac-arm64/VoucherVisionLocal.app/Contents/Resources/uv/uv`
     (if it is not, electron-builder's `mac.binaries` option can list it);
   - both DMGs are stapled (`xcrun stapler validate`);
   - `desktop/build/latest-mac.yml` lists BOTH the arm64 and x64 zips.
5. **Real first run on a clean account.** Install the arm64 DMG into
   `/Applications`, open it as a downloaded file would be (Gatekeeper must not
   complain), and confirm Python setup, a demo run, and on the Engine page
   that the pinned llama.cpp Metal engine installs and passes its self-test.
   This checks that the engine and Python, which the app downloads itself, are
   not blocked by quarantine or by the hardened runtime. Do not download a
   model without asking the owner.
6. **First release.** With the owner's go-ahead: `./deploy.sh` (VERSION is `0.1.0`).
   Then on Linux, the owner (or the Linux agent) can check that the AppImage from
   the release starts.
7. **Update test on the Mac.** Install 0.1.0 from the release. Bump `VERSION`
   to `0.1.1` (any harmless change), commit, push, `./deploy.sh`. Start 0.1.0:
   "Update ready" must appear next to the version number; click it: the app
   must restart as 0.1.1 (check the version label). Repeat with 0.1.2 and
   quit normally instead: the next start must be 0.1.2. On Apple Silicon,
   confirm the updated app is still arm64 (`file` on the main binary, or
   Activity Monitor's Kind column).
8. **Report back** what you changed (commit on `main`, no attribution
   trailer, push) and anything that still needs the owner.

## Known risks to check

- The Mac updater (Squirrel.Mac) only works for signed apps; an unsigned copy
  shows "This copy is not signed, so it cannot update itself" on the Setup page.
- The entitlements were copied from the Editor. VVL also starts child
  processes (uv, Python, llama-server) and listens on 127.0.0.1; none of that
  needs extra entitlements outside the App Sandbox, but confirm in step 5.
- The app icon (`desktop/build-resources/icon.png`) is a placeholder: the
  Editor's VoucherVision icon upscaled to 1024 px. the owner may want a VVL icon.
- The VoucherVisionGO-Editor has the `latest-mac.yml` bug described above
  (its releases list only the x64 zip). Report it to the owner; do not fix the
  Editor unless he asks.

## Done on the Mac (2026-11-20)

Releases v0.1.0, v0.1.1 and v0.1.2 are on GitHub: Apple Silicon DMG + zip
(signed, notarized, stapled), the Windows NSIS installer (not code-signed, not
run on Windows yet), and the Linux AppImage (not started on Linux yet).

Checked on an M2 Pro, macOS 26.7.1:

- Unsigned smoke test (step 3) and the signed build (step 4): every check above
  passes. The bundled uv is signed with the hardened runtime by electron-builder
  itself, so `mac.binaries` is not needed.
- First run in the owner's account (step 5), launched through LaunchServices
  from a quarantined, translocated copy: Python setup, the Setup page, a demo
  run, and the pinned `macos-arm64-metal` engine b12359 install with its
  self-test. Quit leaves no backend or engine process.
- Updates (step 7): 0.1.0 -> 0.1.1 by "Update ready" (restarted as 0.1.1) and
  0.1.1 -> 0.1.2 by a normal quit (the next start is 0.1.2); still arm64. The
  second update was differential: 3.7 MB of 140 MB, from the zip blockmap now
  in each release. The quit question appeared both times a download was running.

Changed: `deploy.sh` builds the Windows installer as the Editor does, builds
the Mac for Apple Silicon only, checks every updater feed and uploads the
blockmaps; the DMG window shows the app, an arrow and Applications
(`desktop/scripts/make_dmg_background.py`).

Not built: Intel Macs. `pymupdf` 2.1 and `onnxruntime` 1.32.0 (and, in 0.1.x,
`openvino`) have no macOS x86_64 wheels; RELEASING.md says how to bring Intel back.

Found on the way:

- `open -a` from a VS Code terminal passes `ELECTRON_RUN_AS_NODE=1` to the app,
  which then exits at once with no window and no log. Use
  `env -u ELECTRON_RUN_AS_NODE open ...`. Turning off Electron's `runAsNode`
  fuse in the build would make the installed app ignore it.
- A copy that was not moved into Applications by the user runs translocated
  (read-only, random path) and cannot update itself. Electron's
  `app.moveToApplicationsFolder()` could offer the move.
- `latest-mac.yml` lists the DMG's size from before stapling (about 2 KB less);
  the updater reads only the zip entry, so this is harmless.
- The DMG itself is not code-signed (notarized and stapled only, like the
  Editor's); `dmg.sign: true` would sign it.
- Two tests assume a Linux host and fail on a Mac:
  `test_doctor_reports_engine_and_models` and `test_install_route_and_disk_check`.

## Next

The Linux tasks, including the 0.2.0 label detector checks, are in
`docs/release/LINUX_HANDOFF.md`.
