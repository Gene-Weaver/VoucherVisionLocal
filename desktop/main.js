// VoucherVisionLocal desktop shell.
//
// Starts the Python backend on a free local port with a random session token,
// waits for it, shows its UI, and stops it on quit. The token is never logged.
//
// From a source checkout (`npm start`) the backend runs with `uv run vvl serve`.
// In the installed app it runs from the bundled source with the Python
// environment that runtime.js prepares on first launch, and updater.js keeps
// the app up to date.

const { app, BrowserWindow, Menu, dialog, ipcMain, shell, session } = require('electron');
const { spawn, execFile } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const net = require('net');
const os = require('os');
const path = require('path');
const runtime = require('./runtime');
const updater = require('./updater');

const REPO_ROOT = path.resolve(__dirname, '..');
const APP_NAME = 'VoucherVisionLocal';
const DEFAULT_PORT = 8642;
const PORT_ATTEMPTS = 20;
const START_TIMEOUT_MS = 180000;
const TERM_GRACE_MS = 3000;
// On quit the backend first stops the engine (it allows itself 15 seconds), so
// it gets longer than that to exit by itself before it is stopped by force.
const SHUTDOWN_WAIT_MS = 25000;
const OUTPUT_TAIL_LINES = 40;
const TABS = ['vvl', 'models', 'keys', 'prompts', 'engine'];

// ---------------------------------------------------------------- options
const argv = process.argv.slice(1);
const shotFlag = argv.indexOf('--screenshots');
const screenshotDir = shotFlag >= 0
  ? path.resolve(argv[shotFlag + 1] && !argv[shotFlag + 1].startsWith('-') ? argv[shotFlag + 1] : (process.env.VVL_SCREENSHOT_DIR || 'screenshots'))
  : (process.env.VVL_SCREENSHOT_DIR ? path.resolve(process.env.VVL_SCREENSHOT_DIR) : null);
const scenarioFlag = argv.indexOf('--scenario');
const scenarioFile = scenarioFlag >= 0 && argv[scenarioFlag + 1] ? path.resolve(argv[scenarioFlag + 1]) : null;
// Developer aid: show an already-running `vvl serve` (its full URL, token
// included) instead of starting the backend. Must be a loopback http URL.
const externalUiUrl = parseLoopbackUrl(process.env.VVL_UI_URL);
// Developer aid: a separate profile folder, so a test copy can run beside the
// installed app (the single-instance lock is per profile folder).
if (process.env.VVL_USER_DATA_DIR) app.setPath('userData', path.resolve(process.env.VVL_USER_DATA_DIR));
// The installed app keeps Electron's own files beside the backend's data, so
// everything VoucherVisionLocal stores for a user is in one folder (and on
// macOS the two never mix in the same Application Support folder).
else if (app.isPackaged) app.setPath('userData', path.join(runtime.dataHome(), 'electron'));
const PACKAGED = app.isPackaged;

let mainWindow = null;
let backend = null; // {child, port, token, output: string[], exited: boolean}
let appOrigin = null;
let quitState = 'running'; // running | stopping | done
let quitConfirmed = false; // the user agreed to quit while a download or install is running
let askingToQuit = false;
let exitCode = 0;
let restartForUpdate = false; // the user clicked "Update ready"

function parseLoopbackUrl(value) {
  if (!value) return null;
  try {
    const u = new URL(value);
    if (u.protocol === 'http:' && (u.hostname === '127.0.0.1' || u.hostname === 'localhost')) return u;
  } catch (_) { /* ignored: treated as not set */ }
  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------- small HTTP helpers
function httpRequest({ method = 'GET', port, pathName, token = null, timeoutMs = 2000 }) {
  return new Promise((resolve) => {
    const headers = token ? { 'X-VVL-Token': token } : {};
    if (method !== 'GET') headers['Content-Length'] = '0';
    const req = http.request({ host: '127.0.0.1', port, path: pathName, method, headers, timeout: timeoutMs }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        if (body.length < 65536) body += chunk;
      });
      res.on('end', () => {
        let json = null;
        try {
          json = JSON.parse(body);
        } catch (_) { /* not JSON */ }
        resolve({ ok: true, status: res.statusCode, json });
      });
    });
    req.on('timeout', () => req.destroy());
    req.on('error', () => resolve({ ok: false, status: 0, json: null }));
    req.end();
  });
}

function portIsFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.listen({ port, host: '127.0.0.1', exclusive: true }, () => server.close(() => resolve(true)));
  });
}

// ---------------------------------------------------------------- backend process
function findUv() {
  const exe = process.platform === 'win32' ? 'uv.exe' : 'uv';
  const home = os.homedir();
  const candidates = [];
  if (process.env.VVL_UV) candidates.push(process.env.VVL_UV);
  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (dir) candidates.push(path.join(dir, exe));
  }
  candidates.push(
    path.join(home, '.local', 'bin', exe),
    path.join(home, '.cargo', 'bin', exe),
    path.join('/opt/homebrew/bin', exe),
    path.join('/usr/local/bin', exe),
  );
  for (const file of candidates) {
    try {
      fs.accessSync(file, fs.constants.X_OK);
      if (fs.statSync(file).isFile()) return file;
    } catch (_) { /* try the next one */ }
  }
  return null;
}

class StartError extends Error {
  constructor(title, message, detail = '') {
    super(message);
    this.title = title;
    this.detail = detail;
  }
}

function outputTail(proc) {
  // The backend prints its own URL (with the token) once; keep it off the error page.
  return proc.output.slice(-OUTPUT_TAIL_LINES).join('\n').split(proc.token).join('[session token]');
}

function backendCommand(uv, port, token) {
  const ours = { VVL_TOKEN: token, VVL_OWNER_PID: String(process.pid), VVL_PORT: String(port) };
  if (PACKAGED) return runtime.backendCommand(process.resourcesPath, port, ours);
  return {
    command: uv,
    args: ['run', '--project', REPO_ROOT, 'vvl', 'serve', '--port', String(port)],
    cwd: REPO_ROOT,
    env: { ...process.env, ...ours, PYTHONUNBUFFERED: '1' },
  };
}

function spawnBackend(uv, port, token) {
  const cmd = backendCommand(uv, port, token);
  const child = spawn(cmd.command, cmd.args, {
    cwd: cmd.cwd,
    env: cmd.env,
    detached: process.platform !== 'win32', // its own process group, so the whole tree can be stopped
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const proc = { child, port, token, output: [], exited: false, spawnError: null };
  const collect = (chunk) => {
    for (const line of String(chunk).split(/\r?\n/)) {
      if (line.trim()) proc.output.push(line);
    }
    if (proc.output.length > 400) proc.output.splice(0, proc.output.length - 400);
  };
  child.stdout.on('data', collect);
  child.stderr.on('data', collect);
  child.on('error', (err) => {
    proc.spawnError = err;
    proc.exited = true;
  });
  child.on('exit', () => {
    proc.exited = true;
  });
  return proc;
}

function groupAlive(pid) {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

async function killTree(proc) {
  const pid = proc && proc.child && proc.child.pid;
  if (!pid) return;
  if (process.platform === 'win32') {
    await new Promise((resolve) => execFile('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true }, () => resolve()));
    return;
  }
  const signalGroup = (sig) => {
    try {
      process.kill(-pid, sig);
    } catch (_) { /* already gone */ }
  };
  signalGroup('SIGTERM');
  const deadline = Date.now() + TERM_GRACE_MS;
  while (Date.now() < deadline && groupAlive(pid)) await sleep(100);
  if (groupAlive(pid)) {
    signalGroup('SIGKILL');
    const hard = Date.now() + 2000;
    while (Date.now() < hard && groupAlive(pid)) await sleep(50);
  }
}

async function stopBackend() {
  const proc = backend;
  backend = null;
  if (!proc) return;
  if (!proc.exited) {
    // The backend answers at once, stops the engine (which frees graphics
    // memory), and then exits. Signaling it before that finishes would cut the
    // engine stop short, so wait for it to leave by itself.
    const asked = await httpRequest({ method: 'POST', port: proc.port, pathName: '/v1/shutdown', token: proc.token, timeoutMs: 3000 });
    const accepted = asked.ok && asked.status === 200;
    const deadline = Date.now() + (accepted ? SHUTDOWN_WAIT_MS : 1000);
    while (Date.now() < deadline && !proc.exited) await sleep(50);
  }
  await killTree(proc); // nothing to do after a clean exit; otherwise stops whatever is left
}

// ---------------------------------------------------------------- quitting while a job runs
function apiTarget() {
  if (backend && !backend.exited) return { port: backend.port, token: backend.token };
  if (externalUiUrl && externalUiUrl.searchParams.get('token')) {
    return { port: Number(externalUiUrl.port) || 80, token: externalUiUrl.searchParams.get('token') };
  }
  return null;
}

/** Downloads and installs that are running right now (empty when the backend cannot be asked). */
async function runningJobs() {
  const target = apiTarget();
  if (!target) return [];
  const res = await httpRequest({ port: target.port, pathName: '/v1/jobs', token: target.token, timeoutMs: 2500 });
  const jobs = res.ok && res.status === 200 && res.json && Array.isArray(res.json.jobs) ? res.json.jobs : [];
  return jobs.filter((j) => j && ['queued', 'running', 'verifying'].includes(j.state));
}

/** True when it is fine to quit: nothing is running, or the user chose to quit anyway. */
async function confirmQuit({ always = false } = {}) {
  if (!always && (quitConfirmed || screenshotDir)) return true;
  const jobs = await runningJobs();
  if (!jobs.length) return true;
  const download = jobs.find((j) => j.kind === 'model_download' || j.kind === 'detector_download');
  const job = download || jobs[0];
  const percent = job.bytes_total > 0 ? Math.floor((job.bytes_done / job.bytes_total) * 100) : 0;
  const name = String(job.label || (download ? 'A model' : 'The engine'));
  const options = download
    ? {
      message: 'A download is still running',
      detail: `${name} is ${percent}% downloaded. If you quit now, the download stops. It continues from where it stopped the next time you start it.`,
      buttons: ['Keep Downloading', 'Quit'],
    }
    : {
      message: 'The engine is still being installed',
      detail: 'If you quit now, the install stops and has to be started again.',
      buttons: ['Keep Installing', 'Quit'],
    };
  const box = { type: 'question', title: `Quit ${APP_NAME}?`, defaultId: 0, cancelId: 0, noLink: true, ...options };
  const result = mainWindow && !mainWindow.isDestroyed()
    ? await dialog.showMessageBox(mainWindow, box)
    : await dialog.showMessageBox(box);
  return result.response === 1;
}

/** Ask (once at a time) and then run `proceed` if quitting is fine. */
function quitAfterConfirm(proceed) {
  if (askingToQuit) return;
  askingToQuit = true;
  confirmQuit().catch(() => true).then((ok) => {
    askingToQuit = false;
    if (ok) {
      quitConfirmed = true;
      proceed();
    }
  });
}

/** Start the backend on the first usable port. Resolves {port, token}. */
async function startBackend() {
  const uv = PACKAGED ? null : findUv();
  if (!PACKAGED && !uv) {
    throw new StartError('uv is not installed',
      'VoucherVisionLocal uses the uv tool to run its Python backend, and uv was not found on this computer.',
      'Install uv from https://docs.astral.sh/uv/ and start VoucherVisionLocal again.');
  }
  const first = Number(process.env.VVL_PORT) > 0 ? Number(process.env.VVL_PORT) : DEFAULT_PORT;
  let lastDetail = '';
  for (let port = first; port < first + PORT_ATTEMPTS; port += 1) {
    // A port that anything already answers on (another app, or another copy of this one) is skipped.
    if (!(await portIsFree(port))) continue;
    const token = crypto.randomBytes(32).toString('base64url');
    const proc = spawnBackend(uv, port, token);
    backend = proc;
    const deadline = Date.now() + START_TIMEOUT_MS;
    let outcome = 'timeout';
    while (Date.now() < deadline) {
      if (quitState !== 'running') return null;
      if (proc.exited) {
        outcome = 'exited';
        break;
      }
      const health = await httpRequest({ port, pathName: '/health', timeoutMs: 1500 });
      if (health.ok && health.status === 200) {
        if (!health.json || health.json.app !== APP_NAME) {
          outcome = 'foreign';
          break;
        }
        // Make sure it is our server: only it knows this token.
        const check = await httpRequest({ port, pathName: '/v1/app', token, timeoutMs: 5000 });
        outcome = check.ok && check.status === 200 ? 'ready' : 'foreign';
        break;
      }
      await sleep(300);
    }
    if (outcome === 'ready') return { port, token };
    lastDetail = outputTail(proc);
    await killTree(proc);
    backend = null;
    if (outcome === 'foreign') continue; // something else took the port first; try the next one
    if (outcome === 'exited') {
      if (proc.spawnError) throw new StartError('The backend could not be started', `${PACKAGED ? 'Starting Python' : 'Running uv'} failed: ${proc.spawnError.message}`, lastDetail);
      throw new StartError('The backend stopped while starting', 'The VoucherVisionLocal server exited before it was ready.', lastDetail);
    }
    throw new StartError('The backend did not start in time', `The VoucherVisionLocal server did not answer within ${Math.round(START_TIMEOUT_MS / 1000)} seconds.`, lastDetail);
  }
  throw new StartError('No free port', `Ports ${first} to ${first + PORT_ATTEMPTS - 1} are all in use.`, lastDetail);
}

// ---------------------------------------------------------------- Python environment (installed app)
/** Change the start screen's text (loading.html). Best effort. */
function setLoading(win, { title, detail, step } = {}) {
  if (!win || win.isDestroyed()) return;
  const set = (id, value) => (value === undefined ? '' : `document.getElementById(${JSON.stringify(id)}).textContent = ${JSON.stringify(String(value))};`);
  win.webContents.executeJavaScript(`(() => { ${set('title', title)} ${set('detail', detail)} ${set('step', step)} return true; })()`, true).catch(() => {});
}

/** Install or update the Python environment before the first backend start, with progress on the start screen. */
async function prepareRuntime(win) {
  let needed;
  try {
    needed = runtime.needsSync(process.resourcesPath);
  } catch (err) {
    throw new StartError('This copy of VoucherVisionLocal is incomplete', `A file the app needs is missing (${err.message}).`, 'Download and install VoucherVisionLocal again.');
  }
  if (!needed) return;
  const first = !fs.existsSync(runtime.layout(process.resourcesPath).venv);
  setLoading(win, first
    ? { title: 'Setting up VoucherVisionLocal', detail: 'The first start downloads Python and the packages VoucherVisionLocal uses (a few hundred MB). This takes a minute or two.', step: '' }
    : { title: 'Updating VoucherVisionLocal', detail: 'This version uses updated Python packages. This takes a moment.', step: '' });
  try {
    await runtime.sync(process.resourcesPath, (step) => setLoading(win, { step }));
  } catch (err) {
    if (quitState !== 'running') return;
    throw new StartError('Python could not be set up', err.message || 'Setting up Python failed.', err.detail || '');
  }
}

// ---------------------------------------------------------------- updates (installed app)
let updaterStarted = false;

function sendToApp(channel, payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  if (!sameOrigin(mainWindow.webContents.getURL())) return;
  mainWindow.webContents.send(channel, payload);
}

/** Check for an update once per launch, after the app page is showing. */
function startUpdaterOnce() {
  if (updaterStarted || screenshotDir) return;
  updaterStarted = true;
  updater.start({ onState: (s) => sendToApp('vvl:update-state', s), logDir: path.join(runtime.dataHome(), 'logs') });
}

/** Ask before restarting for an update while a batch, download or install is running. */
async function confirmRestartForUpdate(batchRunning) {
  const jobs = await runningJobs();
  if (!batchRunning && !jobs.length) return true;
  const download = jobs.find((j) => j.kind === 'model_download' || j.kind === 'detector_download');
  let message;
  let detail;
  if (batchRunning) {
    message = 'A batch is still being processed';
    detail = 'Restarting now stops the batch. You can also restart later: the update is installed the next time you quit.';
  } else if (download) {
    message = 'A download is still running';
    detail = 'Restarting now pauses the download. It continues from where it stopped the next time you start it.';
  } else {
    message = 'The engine is still being installed';
    detail = 'Restarting now stops the install, and it has to be started again.';
  }
  const box = { type: 'question', title: 'Restart to update?', message, detail, buttons: ['Keep Working', 'Restart Now'], defaultId: 0, cancelId: 0, noLink: true };
  const result = mainWindow && !mainWindow.isDestroyed() ? await dialog.showMessageBox(mainWindow, box) : await dialog.showMessageBox(box);
  return result.response === 1;
}

// ---------------------------------------------------------------- pages
function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function errorPageUrl(err) {
  const title = err.title || 'VoucherVisionLocal could not start';
  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>${APP_NAME}</title><style>
body{margin:0;padding:40px 24px;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#f8f9fa;color:#333}
main{max-width:820px;margin:0 auto;background:#fff;border:1px solid #ddd;border-radius:8px;padding:26px 30px}
h1{font-size:1.4rem;margin:0 0 12px;color:#c62828}p{margin:0 0 12px;line-height:1.5}
pre{background:#111;color:#cfe8cf;padding:12px;border-radius:6px;max-height:320px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:12.5px}
a.btn{display:inline-block;margin-right:10px;padding:8px 16px;border-radius:4px;text-decoration:none;border:1px solid #4caf50;color:#2e7d32}
a.btn.primary{background:#4caf50;color:#fff}
@media (prefers-color-scheme: dark){body{background:#0d0d0d;color:#e0e0e0}main{background:#141414;border-color:#333}h1{color:#ef9a9a}a.btn{color:#81c784}}
</style></head><body><main role="alert">
<h1>${escapeHtml(title)}</h1>
<p>${escapeHtml(err.message || 'An unexpected error occurred.')}</p>
${err.detail ? `<pre>${escapeHtml(err.detail)}</pre>` : ''}
<p><a class="btn primary" href="vvl-action://retry">Try again</a><a class="btn" href="vvl-action://quit">Quit</a></p>
</main></body></html>`;
  return `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;
}

function isHttpsUrl(value) {
  try {
    return new URL(String(value)).protocol === 'https:';
  } catch (_) {
    return false;
  }
}

function sameOrigin(value) {
  if (!appOrigin) return false;
  try {
    return new URL(String(value)).origin === appOrigin;
  } catch (_) {
    return false;
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 1000,
    minHeight: 640,
    show: false, // shown maximized below
    title: APP_NAME,
    backgroundColor: '#f8f9fa',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      additionalArguments: [`--vvl-app-version=${app.getVersion()}`, `--vvl-packaged=${PACKAGED ? 1 : 0}`],
    },
  });

  // External pages never open inside the app: https links go to the system browser, everything else is dropped.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (isHttpsUrl(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (event, url) => {
    let target = null;
    try {
      target = new URL(url);
    } catch (_) { /* blocked below */ }
    if (target && target.protocol === 'vvl-action:') {
      event.preventDefault();
      if (target.hostname === 'retry') launch();
      else if (target.hostname === 'quit') app.quit();
      return;
    }
    if (!sameOrigin(url)) event.preventDefault();
  });
  win.webContents.on('will-attach-webview', (event) => event.preventDefault());
  win.on('close', (event) => {
    // Closing the window quits the app, so a running download gets a question first.
    if (quitState !== 'running' || quitConfirmed || screenshotDir) return;
    event.preventDefault();
    quitAfterConfirm(() => {
      if (!win.isDestroyed()) win.close();
    });
  });
  win.on('closed', () => {
    if (mainWindow === win) mainWindow = null;
  });
  // Open using the whole screen: maximized (macOS: zoomed to the area between the menu bar
  // and the Dock), not full screen. Un-maximizing returns to 1280x860. Screenshot runs keep
  // the fixed size so their images stay comparable.
  if (screenshotDir) {
    win.show();
  } else {
    win.maximize();
    win.show();
  }
  return win;
}

async function launch() {
  if (!mainWindow) mainWindow = createWindow();
  const win = mainWindow;
  try {
    await win.loadFile(path.join(__dirname, 'loading.html'));
    let url;
    if (externalUiUrl) {
      appOrigin = externalUiUrl.origin;
      url = externalUiUrl.href;
    } else {
      await stopBackend();
      if (PACKAGED) await prepareRuntime(win);
      if (quitState !== 'running') return;
      const started = await startBackend();
      if (!started || win.isDestroyed()) return;
      appOrigin = `http://127.0.0.1:${started.port}`;
      url = `${appOrigin}/?token=${encodeURIComponent(started.token)}`;
    }
    await win.loadURL(url);
    startUpdaterOnce();
    if (screenshotDir) {
      await runScreenshots(win).catch((err) => {
        console.error(`[screenshots] failed: ${err && err.message}`);
        exitCode = 1;
        app.quit();
      });
    }
  } catch (err) {
    if (win.isDestroyed() || quitState !== 'running') return;
    const shown = err instanceof StartError ? err : new StartError('VoucherVisionLocal could not start', 'The interface could not be loaded.', '');
    await win.loadURL(errorPageUrl(shown)).catch(() => {});
    if (screenshotDir) {
      console.error(`[screenshots] ${shown.title}: ${shown.message}${err instanceof StartError ? '' : ` (${err && err.message})`}`);
      await sleep(500);
      fs.mkdirSync(screenshotDir, { recursive: true });
      fs.writeFileSync(path.join(screenshotDir, 'startup-error.png'), (await win.webContents.capturePage()).toPNG());
      exitCode = 1;
      app.quit();
    }
  }
}

// ---------------------------------------------------------------- screenshots (developer aid)
async function runScreenshots(win) {
  const wc = win.webContents;
  const problems = [];
  const notes = [];
  const shots = [];
  fs.mkdirSync(screenshotDir, { recursive: true });

  wc.on('console-message', (...args) => {
    const details = args[0] && typeof args[0].level === 'string' ? args[0] : { level: ['verbose', 'info', 'warning', 'error'][args[1]] || 'info', message: args[2], sourceId: args[4], lineNumber: args[3] };
    const text = String(details.message || '');
    // Chromium logs every failed request; the ones a scenario expects are checked by status below.
    const provoked = /Failed to load resource/.test(text) && expected.some((e) => text.includes(String(e.status)));
    if ((details.level === 'error' || details.level === 'warning') && !provoked) {
      problems.push(`console ${details.level}: ${details.message} (${String(details.sourceId || '').replace(/\?.*$/, '')}:${details.lineNumber || 0})`);
    }
  });
  wc.on('render-process-gone', (_e, d) => problems.push(`renderer gone: ${d.reason}`));
  const expected = []; // {status, path}: error answers a scenario provokes on purpose
  wc.session.webRequest.onCompleted((d) => {
    if (d.statusCode >= 400) {
      const u = new URL(d.url);
      const i = expected.findIndex((e) => e.status === d.statusCode && u.pathname.includes(e.path));
      if (i >= 0 && sameOrigin(d.url)) {
        expected.splice(i, 1);
        notes.push(`expected HTTP ${d.statusCode} ${d.method} ${u.pathname}`);
        return;
      }
      (sameOrigin(d.url) ? problems : notes).push(`HTTP ${d.statusCode} ${d.method} ${u.origin}${u.pathname}`);
    }
  });
  wc.session.webRequest.onErrorOccurred((d) => {
    const u = new URL(d.url);
    // A reload or a Stop button ends requests on purpose; that is not a failure.
    (sameOrigin(d.url) && d.error !== 'net::ERR_ABORTED' ? problems : notes).push(`request failed (${d.error}) ${d.method} ${u.origin}${u.pathname}`);
  });

  const js = (code) => wc.executeJavaScript(code, true);
  async function waitFor(code, what, timeoutMs = 30000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await js(code).catch(() => false)) return true;
      await sleep(150);
    }
    problems.push(`timed out waiting for ${what}`);
    return false;
  }
  const idle = (tab) => `Boolean(window.vvlUi && window.vvlUi.activeTab() === ${JSON.stringify(tab)} && !['idle', 'loading'].includes(window.vvlUi.tabState(${JSON.stringify(tab)})) && window.vvlUi.inflight() === 0)`;

  async function shoot(name) {
    await sleep(500);
    // Wait for the page to paint what it has now, so the capture is not an older frame.
    await js('new Promise((resolve) => { setTimeout(() => resolve(false), 1500); requestAnimationFrame(() => requestAnimationFrame(() => resolve(true))); })').catch(() => {});
    const file = path.join(screenshotDir, `${name}.png`);
    fs.writeFileSync(file, (await wc.capturePage()).toPNG());
    shots.push(file);
    try {
      // The whole page, not just the window's visible part.
      const metrics = await wc.debugger.sendCommand('Page.getLayoutMetrics');
      const size = metrics.cssContentSize || metrics.contentSize;
      const full = await wc.debugger.sendCommand('Page.captureScreenshot', {
        format: 'png', captureBeyondViewport: true,
        clip: { x: 0, y: 0, width: Math.ceil(size.width), height: Math.min(Math.ceil(size.height), 12000), scale: 1 },
      });
      const fullFile = path.join(screenshotDir, `${name}-full.png`);
      fs.writeFileSync(fullFile, Buffer.from(full.data, 'base64'));
      shots.push(fullFile);
    } catch (err) {
      notes.push(`full-page capture failed for ${name}: ${err.message}`);
    }
  }

  try {
    wc.debugger.attach('1.3');
  } catch (err) {
    notes.push(`debugger not attached: ${err.message}`);
  }

  const shellReady = await waitFor(`Boolean(window.vvlUi && document.getElementById('tab-vvl'))`, 'the app shell');
  if (shellReady && scenarioFile) {
    // A scenario is a module exporting async (tools) => {}; see desktop/scenarios/.
    try {
      await require(scenarioFile)({
        js, waitFor, shoot, sleep, idle, problems, notes, win, wc, fs, path, REPO_ROOT, screenshotDir,
        expectHttp: (status, pathPart) => expected.push({ status, path: pathPart }),
        // The question asked when quitting during a download (a screenshot run normally skips it).
        confirmQuit: () => confirmQuit({ always: true }),
        setTheme: (theme) => js(`if (document.documentElement.dataset.theme !== ${JSON.stringify(theme)}) document.getElementById('themeToggle').click(); true`),
        gotoTab: async (tab) => {
          await js(`document.getElementById('tab-${tab}').click(); true`);
          await waitFor(idle(tab), `the ${tab} tab`);
        },
      });
    } catch (err) {
      problems.push(`scenario failed: ${err && err.stack ? err.stack : err}`);
    }
  } else if (shellReady) {
    for (const theme of ['light', 'dark']) {
      await js(`if (document.documentElement.dataset.theme !== ${JSON.stringify(theme)}) document.getElementById('themeToggle').click(); true`);
      for (const tab of TABS) {
        await js(`document.getElementById('tab-${tab}').click(); true`);
        await waitFor(idle(tab), `the ${tab} tab`);
        await js('window.scrollTo(0, 0); true');
        await shoot(`${tab}-${theme}`);
      }
      // One demo run on the VVL tab, with a sample image from the repository.
      const sample = path.join(REPO_ROOT, 'tests', 'fixtures', 'images', 'MICH_16205594_Poaceae_Jouvea_pilosa.jpg');
      if (fs.existsSync(sample)) {
        const b64 = fs.readFileSync(sample).toString('base64');
        await js(`document.getElementById('tab-vvl').click(); true`);
        await waitFor(idle('vvl'), 'the vvl tab');
        await js(`(async () => {
          // This tour always uses the demo engine, whatever model was chosen last.
          for (const name of ['engines', 'llm_model']) {
            const demo = document.querySelector('input[name="' + name + '"][value="demo:mock"]');
            if (demo && !demo.checked) demo.click();
          }
          for (const box of document.querySelectorAll('input[name="engines"]:checked')) if (box.value !== 'demo:mock') box.click();
          const bytes = Uint8Array.from(atob(${JSON.stringify(b64)}), (c) => c.charCodeAt(0));
          const dt = new DataTransfer();
          dt.items.add(new File([bytes], 'MICH_16205594_Poaceae_Jouvea_pilosa.jpg', { type: 'image/jpeg' }));
          const input = document.getElementById('fileInput');
          input.files = dt.files;
          input.dispatchEvent(new Event('change', { bubbles: true }));
          const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
          tabs.find((t) => t.dataset.tab === 'file').click();
          input.closest('[role="tabpanel"]').querySelector('button.button').click();
          return true;
        })()`);
        const finished = await waitFor(`Boolean(document.querySelector('.split-results-container') || document.querySelector('.results .state-error'))`, 'a demo result', 120000);
        if (finished && await js(`Boolean(document.querySelector('.results .state-error'))`)) {
          problems.push(`demo run failed: ${await js(`document.querySelector('.results .state-error').textContent`)}`);
        }
        await sleep(2500); // let map tiles arrive
        await js(`document.querySelector('.step-container:last-of-type').scrollIntoView(); true`);
        await shoot(`vvl-result-${theme}`);
      } else {
        notes.push('sample image not found; no demo run');
      }
    }
  }

  const report = { ok: problems.length === 0, problems, notes, screenshots: shots };
  fs.writeFileSync(path.join(screenshotDir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`[screenshots] ${shots.length} images in ${screenshotDir}; ${problems.length} problem(s)`);
  for (const p of problems) console.log(`[screenshots] problem: ${p}`);
  exitCode = problems.length ? 1 : 0;
  app.quit();
}

// ---------------------------------------------------------------- app lifecycle
function buildMenu() {
  const template = [
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    { label: 'File', submenu: [{ role: process.platform === 'darwin' ? 'close' : 'quit' }] },
    { role: 'editMenu' },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }, { type: 'separator' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { type: 'separator' }, { role: 'togglefullscreen' }] },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function fromApp(event) {
  return Boolean(event.senderFrame && sameOrigin(event.senderFrame.url));
}

/**
 * The single-instance lock. Right after an update the new version can start
 * while the old one is still closing, so it waits a little for the lock then.
 */
async function acquireInstanceLock() {
  const afterUpdate = process.env.VVL_AFTER_UPDATE === '1';
  delete process.env.VVL_AFTER_UPDATE;
  if (app.requestSingleInstanceLock()) return true;
  if (!afterUpdate) return false;
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    await sleep(250);
    if (app.requestSingleInstanceLock()) return true;
  }
  return false;
}

function setupApp() {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  ipcMain.handle('vvl:open-external', async (event, url) => {
    if (!fromApp(event) || !isHttpsUrl(url)) return false;
    await shell.openExternal(new URL(url).href);
    return true;
  });
  ipcMain.handle('vvl:pick-folder', async (event) => {
    if (!fromApp(event) || !mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, { title: 'Choose a folder', properties: ['openDirectory', 'createDirectory'] });
    return result.canceled || !result.filePaths.length ? null : result.filePaths[0];
  });
  ipcMain.handle('vvl:open-folder', async (event, folder) => {
    // Only an existing folder, given by its full path (a results folder).
    if (!fromApp(event) || typeof folder !== 'string' || !path.isAbsolute(folder)) return false;
    try {
      if (!fs.statSync(folder).isDirectory()) return false;
    } catch (_) {
      return false;
    }
    return (await shell.openPath(folder)) === '';
  });
  ipcMain.handle('vvl:update-state', (event) => (fromApp(event) ? updater.getState() : null));
  ipcMain.handle('vvl:update-install', async (event, options) => {
    if (!fromApp(event) || !updater.isReady() || quitState !== 'running') return false;
    if (!(await confirmRestartForUpdate(Boolean(options && options.batchRunning)))) return false;
    restartForUpdate = true;
    quitConfirmed = true;
    app.quit();
    return true;
  });

  app.whenReady().then(() => {
    // The page needs no camera, microphone, location or notifications.
    session.defaultSession.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
    buildMenu();
    launch();
  });

  app.on('window-all-closed', () => app.quit());

  app.on('before-quit', (event) => {
    if (quitState === 'done') return;
    event.preventDefault();
    if (quitState === 'stopping') return;
    if (!quitConfirmed && !screenshotDir) {
      quitAfterConfirm(() => app.quit());
      return;
    }
    quitState = 'stopping';
    runtime.cancel();
    stopBackend().catch(() => {}).finally(() => {
      quitState = 'done';
      // A downloaded update is installed now: right away for "Update ready",
      // otherwise as part of this normal quit.
      if (exitCode === 0 && updater.isReady()) {
        if (restartForUpdate) process.env.VVL_AFTER_UPDATE = '1';
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.hide();
        if (updater.installAndQuit({ relaunch: restartForUpdate })) {
          // On macOS the updater quits the app itself once the update is in place.
          if (process.platform === 'darwin') {
            setTimeout(() => app.exit(0), 60000);
            return;
          }
        }
      }
      app.exit(exitCode);
    });
  });

  // A signal is not a person at the keyboard: quit without asking.
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(sig, () => {
      quitConfirmed = true;
      app.quit();
    });
  }
}

acquireInstanceLock().then((locked) => {
  if (locked) setupApp();
  else app.quit();
});
