// The Python side of the installed app.
//
// The installer carries the backend source (resources/backend), the files uv
// needs (resources/python: pyproject.toml, uv.lock, python-version) and one
// pinned uv binary (resources/uv). On first launch, and again only when an
// update changes uv.lock, the pinned uv installs the pinned Python and the
// locked packages into the user's data folder:
//
//   <data>/runtime/python/    Python itself (uv-managed, never the system Python)
//   <data>/runtime/venv/      the packages from uv.lock
//   <data>/runtime/uv-cache/  uv's download cache (lets updates reuse packages)
//   <data>/runtime/pycache/   compiled .pyc files (never written into the app)
//   <data>/runtime/runtime.json  what was installed, so the next start can skip it
//
// <data> is the backend's own data folder (platformdirs user_data_dir), so the
// whole install lives in one folder per user.
const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const APP_DIR_NAME = 'VoucherVisionLocal';
const SYNC_TIMEOUT_MS = 30 * 60 * 1000;
let current = null; // the running uv process, if any

/** The backend's data folder; must match vvl/paths.py home(). */
function dataHome() {
  if (process.env.VVL_HOME && process.env.VVL_HOME.trim()) return path.resolve(process.env.VVL_HOME.trim());
  const home = os.homedir();
  if (process.platform === 'darwin') return path.join(home, 'Library', 'Application Support', APP_DIR_NAME);
  if (process.platform === 'win32') return path.join(process.env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), APP_DIR_NAME);
  const xdg = process.env.XDG_DATA_HOME && path.isAbsolute(process.env.XDG_DATA_HOME) ? process.env.XDG_DATA_HOME : path.join(home, '.local', 'share');
  return path.join(xdg, APP_DIR_NAME);
}

function layout(resourcesPath) {
  const runtime = path.join(dataHome(), 'runtime');
  const win = process.platform === 'win32';
  return {
    backendDir: path.join(resourcesPath, 'backend'),
    projectDir: path.join(resourcesPath, 'python'),
    uv: path.join(resourcesPath, 'uv', win ? 'uv.exe' : 'uv'),
    uvVersionFile: path.join(resourcesPath, 'uv', 'VERSION'),
    runtime,
    venv: path.join(runtime, 'venv'),
    python: path.join(runtime, 'venv', win ? path.join('Scripts', 'python.exe') : path.join('bin', 'python')),
    pythonInstalls: path.join(runtime, 'python'),
    cache: path.join(runtime, 'uv-cache'),
    pycache: path.join(runtime, 'pycache'),
    stateFile: path.join(runtime, 'runtime.json'),
  };
}

function readText(file) {
  return fs.readFileSync(file, 'utf8').trim();
}

/** What the installed environment must match: uv.lock, the Python version and the uv version. */
function wanted(l) {
  const lock = fs.readFileSync(path.join(l.projectDir, 'uv.lock'));
  const python = readText(path.join(l.projectDir, 'python-version'));
  const uv = fs.existsSync(l.uvVersionFile) ? readText(l.uvVersionFile) : 'unknown';
  const key = crypto.createHash('sha256').update(lock).update(`\npython=${python}\nuv=${uv}\n`).digest('hex');
  return { key, python, uv };
}

function readState(l) {
  try {
    return JSON.parse(fs.readFileSync(l.stateFile, 'utf8'));
  } catch (_) {
    return null;
  }
}

/** True when the environment is missing or was made for a different uv.lock / Python / uv. */
function needsSync(resourcesPath) {
  const l = layout(resourcesPath);
  const state = readState(l);
  return !(state && state.key === wanted(l).key && fs.existsSync(l.python));
}

/** A clean environment for uv and Python: nothing from the user's own Python or uv setup leaks in. */
function cleanEnv(extra) {
  const env = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith('UV_') || k.startsWith('PYTHON') || k === 'VIRTUAL_ENV' || k === 'CONDA_PREFIX' || k === 'ELECTRON_RUN_AS_NODE') continue;
    env[k] = v;
  }
  return { ...env, ...extra };
}

/** Turns one uv output line into a short message for the start screen, or null. */
function describeLine(line, python) {
  const text = line.trim();
  if (!text) return null;
  if (/cpython|Installed Python|Downloading Python/i.test(text)) return `Downloading Python ${python}`;
  let m = text.match(/^Downloading (\S+)/);
  if (m) return `Downloading ${m[1]}`;
  m = text.match(/^Prepared (\d+) packages?/);
  if (m) return `Prepared ${m[1]} packages`;
  m = text.match(/^Installed (\d+) packages?/);
  if (m) return `Installed ${m[1]} packages`;
  if (/^(Using|Creating virtual environment)/.test(text)) return 'Creating the Python environment';
  return null;
}

/**
 * Install or update the Python environment. Calls onProgress(message) with
 * short messages. Resolves the new state; rejects with {message, detail}.
 */
function sync(resourcesPath, onProgress = () => {}) {
  const l = layout(resourcesPath);
  const want = wanted(l);
  fs.mkdirSync(l.runtime, { recursive: true });
  // A half-finished earlier attempt must not count as installed.
  fs.rmSync(l.stateFile, { force: true });

  const args = ['sync', '--frozen', '--no-dev', '--no-install-project', '--python', want.python, '--project', l.projectDir];
  const env = cleanEnv({
    UV_PROJECT_ENVIRONMENT: l.venv,
    UV_PYTHON_INSTALL_DIR: l.pythonInstalls,
    UV_CACHE_DIR: l.cache,
    UV_PYTHON_PREFERENCE: 'only-managed',
    UV_PYTHON_DOWNLOADS: 'automatic',
    UV_NO_CONFIG: '1',
    UV_NO_PROGRESS: '1',
  });
  onProgress(`Setting up Python ${want.python}`);
  return new Promise((resolve, reject) => {
    const output = [];
    let child;
    try {
      child = spawn(l.uv, args, { cwd: l.projectDir, env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) {
      reject({ message: `The bundled uv could not be started: ${err.message}`, detail: '' });
      return;
    }
    current = child;
    const timer = setTimeout(() => child.kill(), SYNC_TIMEOUT_MS);
    const collect = (chunk) => {
      for (const line of String(chunk).split(/\r?\n/)) {
        if (!line.trim()) continue;
        output.push(line);
        if (output.length > 200) output.shift();
        const msg = describeLine(line, want.python);
        if (msg) onProgress(msg);
      }
    };
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);
    child.on('error', (err) => {
      clearTimeout(timer);
      current = null;
      reject({ message: `The bundled uv could not be started: ${err.message}`, detail: output.join('\n') });
    });
    child.on('exit', (code, signal) => {
      clearTimeout(timer);
      current = null;
      if (code === 0 && fs.existsSync(l.python)) {
        const state = { key: want.key, python: want.python, uv: want.uv, installed_at: new Date().toISOString() };
        fs.writeFileSync(l.stateFile, `${JSON.stringify(state, null, 2)}\n`);
        resolve(state);
        return;
      }
      const tail = output.slice(-40).join('\n');
      const offline = /dns|resolve|connect|network|timed out|offline|tls|certificate/i.test(tail);
      reject({
        message: offline
          ? 'VoucherVisionLocal could not download Python and its packages. The first start needs an internet connection (about 300 MB).'
          : `Setting up Python failed (${signal ? `stopped by ${signal}` : `exit code ${code}`}).`,
        detail: tail,
      });
    });
  });
}

/** Command, arguments and environment that start the backend in the installed app. */
function backendCommand(resourcesPath, port, extraEnv) {
  const l = layout(resourcesPath);
  return {
    command: l.python,
    args: ['-m', 'vvl.cli', 'serve', '--port', String(port)],
    cwd: l.backendDir,
    env: cleanEnv({
      ...extraEnv,
      PYTHONPATH: l.backendDir,
      PYTHONPYCACHEPREFIX: l.pycache,
      PYTHONNOUSERSITE: '1',
      PYTHONUNBUFFERED: '1',
      VVL_HOME: dataHome(),
      VVL_PACKAGED: '1',
    }),
  };
}

/** Stop a running setup (the app is quitting). The next start begins it again. */
function cancel() {
  if (current && !current.killed) current.kill();
}

module.exports = { dataHome, layout, needsSync, sync, cancel, backendCommand };
