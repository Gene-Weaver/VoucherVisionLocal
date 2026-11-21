# Building and releasing VoucherVisionLocal

The desktop app is built and released the same way as VoucherVisionGO-Editor:
`./deploy.sh` runs on a Mac, builds every platform, notarizes the Mac builds,
and creates a GitHub release. Installed apps update themselves from the
latest GitHub release.

## What a release contains

| File | Who uses it |
|---|---|
| `VoucherVisionLocal-<v>-arm64.dmg` | people installing on macOS (Apple Silicon) |
| `VoucherVisionLocal-<v>-arm64.zip` (+ `.blockmap`) | the macOS updater |
| `VoucherVisionLocal-Setup-<v>-x64.exe` (+ `.blockmap`) | Windows: the installer, and the Windows updater |
| `VoucherVisionLocal-<v>-x86_64.AppImage` | Linux: download, `chmod +x`, run; it also updates itself |
| `latest-mac.yml`, `latest.yml` (Windows), `latest-linux.yml` | the updater: newest version, file names, sha512 sums |

The `.blockmap` files let the updaters download only the parts that changed.

**No Intel Mac build.** `onnxruntime` 1.32.0, `openvino` 2026.4.1 and
`pymupdf` 2.1 in `uv.lock` have no macOS x86_64 wheels (the first two have no
source package either), so `uv sync` fails on an Intel Mac and the first-launch
setup cannot finish. Checked under Rosetta on 2026-11-20. To bring Intel back:
add `sys_platform == 'darwin' and platform_machine == 'x86_64'` to
`tool.uv.required-environments` in `pyproject.toml` (uv then locks versions
that have Intel wheels, or says none exist), re-lock, add `x64` to the `mac`
targets in `desktop/package.json`, and build both Mac architectures in one
`--mac --arm64 --x64` run in `deploy.sh` so `latest-mac.yml` lists both zips.

**Windows is not code-signed** (neither is the Editor's). SmartScreen shows
"Windows protected your PC" until the user clicks More info, then Run anyway.
Updates still work. To sign, put `WIN_CSC_LINK` (the .pfx file) and
`WIN_CSC_KEY_PASSWORD` in `.env.signing`; electron-builder picks them up.
The Windows build is made on the Mac and has not been run on Windows yet.

## Versions

`VERSION` (repo root) is the only version number. `npm run stage` copies it
into `desktop/package.json` and `package-lock.json`; the Python backend reads
it directly. The git tag is `v<VERSION>`.

The pinned pieces move only with a release:

- Python packages: `uv.lock`. Python itself: `.python-version`.
- uv (bundled in the installer): `desktop/bundled-uv.json` (version + sha256 per target).
- llama.cpp: `vvl/llama/engine.lock.yaml`. The Setup page offers an engine
  update when an installed engine differs from the pinned one.
- Electron, electron-builder, electron-updater: exact versions in `desktop/package.json`.

## Releasing (on the Mac)

1. Set the new version in `VERSION`, commit, push to `main`.
2. `./deploy.sh`

Build from a git clone outside any synced folder (Dropbox, iCloud Drive). A
synced `node_modules`, `desktop/build` or `.git` is shared with the other
computers, and a Mac build there replaces their Linux Electron and can leave
their git objects half-copied.

`deploy.sh` refuses to run unless the working tree is clean and `main` equals
`origin/main`, and it refuses to release Mac builds that are not signed and
accepted by Gatekeeper. Signing secrets come from `.env.signing` (gitignored;
template in `.env.signing.example`). `./deploy.sh --skip-release` builds
without publishing; add `--allow-unsigned` for a local unsigned test build.

Re-running `deploy.sh` for a version that is already released deletes that
release and its tag first (same as the Editor).

## What the installed app does

- **Installer contents.** Electron shell (`desktop/main.js`, `preload.js`,
  `runtime.js`, `updater.js`), the backend source (`resources/backend`: no
  `pyproject.toml`, so the backend keeps user files in its data folder, never
  in the app), the uv project files (`resources/python`) and the pinned uv
  (`resources/uv`). Built by `desktop/scripts/stage.js`.
- **First launch.** `runtime.js` runs the bundled uv: `uv sync --frozen
  --no-dev --no-install-project --python <pinned>` into
  `<data>/runtime/venv` with uv-managed Python in `<data>/runtime/python`.
  The start screen shows progress. It runs again only when `uv.lock`, the
  Python version, or the uv version changes (`<data>/runtime/runtime.json`).
- **Data folder** (`<data>`): Linux `~/.local/share/VoucherVisionLocal`,
  macOS `~/Library/Application Support/VoucherVisionLocal`, Windows
  `%LOCALAPPDATA%\VoucherVisionLocal`. Electron's own profile is in
  `<data>/electron`. Models stay in the user's chosen models folder.
- **Updates.** One check at launch. A newer release downloads quietly; an
  "Update ready" button appears next to the version number. Clicking it
  asks first if a batch, download or engine install is running, then
  restarts into the new version. Otherwise the update is installed the next
  time the app quits. Off in `npm start` runs and with `VVL_NO_UPDATE=1`.
  Log: `<data>/logs/updater.log`.

## Building and testing on Linux

```bash
cd desktop
npm run dist                     # stage + AppImage in desktop/build/
```

Run a test copy without touching your real data:

```bash
VVL_HOME=/some/scratch/home VVL_USER_DATA_DIR=/some/scratch/profile VVL_PORT=8694 \
  ./build/VoucherVisionLocal-<v>-x86_64.AppImage
```

Test an update locally: build a higher version into a folder (set `VERSION`,
`npm run dist`, copy the new AppImage and `latest-linux.yml`, restore
`VERSION`), serve it with `python3 -m http.server 8697 --bind 127.0.0.1`, and
start the older AppImage with `VVL_UPDATE_URL=http://127.0.0.1:8697/` (only
loopback http URLs are accepted). Both paths were verified this way on
2026-11-20: "Update ready" (replaced the file and relaunched the new version)
and install-on-quit (replaced the file, no relaunch).

Ubuntu 24.04 and newer do not install `libfuse2` by default, which AppImages
need: `sudo apt install libfuse2t64`. (electron-builder's FUSE-free AppImage
runtime is still marked beta; the Editor uses the same classic runtime.)
