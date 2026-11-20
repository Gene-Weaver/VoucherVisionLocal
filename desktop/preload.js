// The only bridge between the VoucherVisionLocal page and the desktop shell.
const { contextBridge, ipcRenderer } = require('electron');

function arg(name) {
  const found = process.argv.find((a) => a.startsWith(`--${name}=`));
  return found ? found.slice(name.length + 3) : '';
}

contextBridge.exposeInMainWorld('vvlDesktop', {
  /** Open an https page in the system browser. Resolves true when it was opened. */
  openExternal: (url) => ipcRenderer.invoke('vvl:open-external', String(url)),
  /** Ask the user for a folder. Resolves its path, or null when canceled. */
  pickFolder: () => ipcRenderer.invoke('vvl:pick-folder'),
  platform: process.platform,
  appVersion: arg('vvl-app-version'),
  /** True in the installed app (false when run from a source checkout). */
  packaged: arg('vvl-packaged') === '1',
  update: {
    /** Resolves {status, version, percent, error, reason}; see updater.js. */
    getState: () => ipcRenderer.invoke('vvl:update-state'),
    /** Calls cb(state) on every change. Returns a function that stops it. */
    onState: (cb) => {
      const handler = (_event, state) => cb(state);
      ipcRenderer.on('vvl:update-state', handler);
      return () => ipcRenderer.removeListener('vvl:update-state', handler);
    },
    /** Restart into the downloaded update. Resolves false if the user chose to keep working. */
    install: (options) => ipcRenderer.invoke('vvl:update-install', { batchRunning: Boolean(options && options.batchRunning) }),
  },
});
