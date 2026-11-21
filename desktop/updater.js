// Auto-update for the installed app (electron-updater + GitHub Releases).
//
// Checks once, at launch. A newer release downloads quietly; when it is ready
// the page shows "Update ready" next to the version number. Clicking it
// restarts into the new version; otherwise the update is installed the next
// time the app quits.
//
// Off in development runs (`npm start`), when VVL_NO_UPDATE=1, and on Linux
// unless the app runs as an AppImage. For testing, VVL_UPDATE_URL may point at
// a local folder served over http://127.0.0.1 that holds a build's
// latest-*.yml and installer files.
const { app } = require('electron');
const fs = require('fs');
const path = require('path');

let autoUpdater = null;
let state = { status: 'disabled', version: null, percent: null, error: null, reason: 'not started' };
let listener = () => {};
let logFile = null;

function setState(patch) {
  state = { ...state, ...patch };
  try {
    listener({ ...state });
  } catch (_) { /* the window may be gone */ }
}

function log(level, args) {
  if (!logFile) return;
  const text = args.map((a) => (a instanceof Error ? a.stack || a.message : typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
  try {
    fs.appendFileSync(logFile, `${new Date().toISOString()} ${level} ${text}\n`);
  } catch (_) { /* logging is best effort */ }
}

const logger = {
  info: (...a) => log('info', a),
  warn: (...a) => log('warn', a),
  error: (...a) => log('error', a),
  debug: () => {},
};

function loopbackHttpUrl(value) {
  if (!value) return null;
  try {
    const u = new URL(value);
    if (u.protocol === 'http:' && (u.hostname === '127.0.0.1' || u.hostname === 'localhost')) return u.href;
  } catch (_) { /* ignored */ }
  return null;
}

/** A short, readable reason for a failed check (the full error goes to the log). */
function describeError(err) {
  const text = String((err && err.message) || err || '');
  if (/404|Cannot find latest|No published versions/i.test(text)) return 'No release has been published yet.';
  if (/ENOTFOUND|EAI_AGAIN|ECONNREFUSED|ETIMEDOUT|ENETUNREACH|net::/i.test(text)) return 'Could not reach GitHub to check for updates.';
  if (/code signature|signed/i.test(text)) return 'This copy is not signed, so it cannot update itself.';
  return text.split('\n')[0].slice(0, 200) || 'The update check failed.';
}

/**
 * Start the updater. onState(state) is called on every change.
 * Returns the initial state.
 */
function start({ onState, logDir }) {
  listener = onState || (() => {});
  if (logDir) {
    try {
      fs.mkdirSync(logDir, { recursive: true });
      logFile = path.join(logDir, 'updater.log');
    } catch (_) { /* no log */ }
  }
  const testFeed = loopbackHttpUrl(process.env.VVL_UPDATE_URL);
  let reason = null;
  if (!app.isPackaged) reason = 'development build';
  else if (process.env.VVL_NO_UPDATE === '1') reason = 'turned off (VVL_NO_UPDATE=1)';
  else if (process.platform === 'linux' && !process.env.APPIMAGE) reason = 'only the AppImage updates itself';
  if (reason) {
    setState({ status: 'disabled', reason });
    return { ...state };
  }

  ({ autoUpdater } = require('electron-updater'));
  autoUpdater.logger = logger;
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.allowPrerelease = false;
  autoUpdater.allowDowngrade = false;
  if (testFeed) {
    autoUpdater.setFeedURL({ provider: 'generic', url: testFeed });
    logger.info(`test feed ${testFeed}`);
  }

  autoUpdater.on('checking-for-update', () => setState({ status: 'checking', error: null, reason: null }));
  autoUpdater.on('update-not-available', () => setState({ status: 'none', version: null, percent: null }));
  autoUpdater.on('update-available', (info) => setState({ status: 'downloading', version: info && info.version, percent: 0 }));
  autoUpdater.on('download-progress', (p) => setState({ status: 'downloading', percent: Math.floor((p && p.percent) || 0) }));
  autoUpdater.on('update-downloaded', (info) => setState({ status: 'ready', version: info && info.version, percent: 100 }));
  autoUpdater.on('error', (err) => {
    // electron-updater logs the error itself.
    // A failure after the download finished does not take the ready update away.
    if (state.status !== 'ready') setState({ status: 'error', error: describeError(err) });
  });

  setState({ status: 'checking', reason: null });
  autoUpdater.checkForUpdates().catch((err) => {
    if (state.status !== 'ready') setState({ status: 'error', error: describeError(err) });
  });
  return { ...state };
}

function getState() {
  return { ...state };
}

function isReady() {
  return Boolean(autoUpdater) && state.status === 'ready';
}

/**
 * Install the downloaded update while the app quits. `relaunch` starts the new
 * version afterwards (the "Update ready" button); without it the app just
 * closes (a normal quit). Returns true when the install was handed off and
 * the updater now ends the app itself.
 */
function installAndQuit({ relaunch }) {
  if (!isReady()) return false;
  try {
    if (process.platform === 'darwin') {
      // On a normal quit Squirrel installs the update after the app has exited.
      if (!relaunch) return false;
      autoUpdater.quitAndInstall();
      return true;
    }
    // Linux AppImage and Windows (NSIS): replace the app now, then quit.
    const ok = autoUpdater.install(true, Boolean(relaunch));
    return Boolean(ok);
  } catch (err) {
    logger.error(err);
    return false;
  }
}

module.exports = { start, getState, isReady, installAndQuit };
