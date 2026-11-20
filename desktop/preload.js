// The only bridge between the VoucherVisionLocal page and the desktop shell.
const { contextBridge, ipcRenderer } = require('electron');

const versionArg = process.argv.find((a) => a.startsWith('--vvl-app-version='));

contextBridge.exposeInMainWorld('vvlDesktop', {
  /** Open an https page in the system browser. Resolves true when it was opened. */
  openExternal: (url) => ipcRenderer.invoke('vvl:open-external', String(url)),
  /** Ask the user for a folder. Resolves its path, or null when canceled. */
  pickFolder: () => ipcRenderer.invoke('vvl:pick-folder'),
  platform: process.platform,
  appVersion: versionArg ? versionArg.slice('--vvl-app-version='.length) : '',
});
