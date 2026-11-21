// Screenshot scenario for the Setup page (ui/js/tabs/setup.js): first run, the
// three "How will you run models?" choices, the engine install and model download
// steps, and the "Update ready" button in the header.
//
// Run it against tools/dev_fake_backend.py with a fresh --root, never against a
// real models folder. Nothing here calls OpenRouter: the key is saved without
// testing it, and the models are added through settings.
//
//   uv run python tools/dev_fake_backend.py --root <scratch> --port 8652 --files-port 8657 --rate 2
//   VVL_UI_URL='http://127.0.0.1:8652/?token=dev-fake-token' VVL_USER_DATA_DIR=<scratch>/electron \
//     npm run screenshots -- <outDir> --scenario scenarios/setup-page.js
//
// The desktop bridge (window.vvlDesktop) is frozen, so the shell's updater cannot be
// swapped from here. The page takes a stand-in through window.vvlUi.useUpdater,
// which this scenario uses to show the update states.

// The fake updater. `window.__fakeUpdate.set({...})` changes its state.
const FAKE_UPDATER = `(() => {
  let state = { status: 'none', version: null, percent: null, error: null, reason: null };
  const subs = new Set();
  const update = {
    getState: async () => state,
    onState: (cb) => { subs.add(cb); return () => subs.delete(cb); },
    install: async (opts) => { window.__fakeUpdate.installs.push(opts); return true; },
  };
  window.__fakeUpdate = { installs: [], set(next) { state = { ...state, ...next }; for (const cb of subs) cb(state); } };
  window.vvlUi.useUpdater(update);
  return true;
})()`;

module.exports = async function setupPage(t) {
  const { js, waitFor, shoot, sleep, problems, notes, setTheme, win } = t;
  const must = async (code, what) => {
    const ok = await js(code).catch(() => false);
    if (!ok) problems.push(`check failed: ${what}`);
    return ok;
  };
  const authFetch = (method, pathName, body) => js(`fetch(${JSON.stringify(pathName)}, {method: ${JSON.stringify(method)},
    headers: {'X-VVL-Token': sessionStorage.getItem('vvl.session-token'), 'Content-Type': 'application/json'},
    body: ${body === undefined ? 'undefined' : JSON.stringify(JSON.stringify(body))}}).then((r) => r.json())`);
  const setupReady = () => waitFor(`window.vvlUi.activeTab() === 'setup' && window.vvlUi.tabState('setup') === 'ready' && window.vvlUi.inflight() === 0`, 'the Setup page', 30000);
  const choose = async (usage) => {
    await js(`document.querySelector('.setup-choice[data-usage="${usage}"]').click(); true`);
    await waitFor(`document.querySelector('.setup-choice[data-usage="${usage}"]').getAttribute('aria-checked') === 'true' && !document.querySelector('.setup-choice:disabled') && window.vvlUi.inflight() === 0`, `the ${usage} choice`, 30000);
    await sleep(300);
  };
  const stepText = (id) => `(document.querySelector('.setup-item[data-step="${id}"]') || {}).textContent || ''`;
  const stepIs = (id, status) => `Boolean(document.querySelector('.setup-item[data-step="${id}"].is-${status}'))`;
  const clickIn = (id, text) => must(`(() => { const b = [...document.querySelectorAll('.setup-item[data-step="${id}"] button')].find((n) => n.textContent.trim().startsWith(${JSON.stringify(text)})); if (!b || b.disabled) return false; b.click(); return true; })()`, `click ${text} in ${id}`);
  const bothThemes = async (name, before = null) => {
    for (const theme of ['light', 'dark']) {
      await setTheme(theme);
      if (before) await before();
      await js('window.scrollTo(0, 0); true');
      await shoot(`${name}-${theme}`);
    }
  };
  const resize = async (width, height) => {
    if (win.isMaximized()) win.unmaximize();
    win.setContentSize(width, height);
    await waitFor(`window.innerWidth === ${width}`, `a ${width}px page`, 5000);
    await sleep(300);
  };
  await resize(1400, 1000);

  // ------------------------------------------------------------ first run
  // A fresh scratch backend: no choice, no key, no engine. The page opens on Setup.
  await waitFor(`window.vvlUi.activeTab() !== null`, 'a tab to open');
  await must(`window.vvlUi.activeTab() === 'setup'`, 'first run opens the Setup page');
  await must(`location.hash === '#setup'`, 'first run sets #setup');
  await setupReady();
  await must(`document.getElementById('updateReady').hidden`, 'no update pill without an updater');
  await must(`/development builds|desktop app/.test(document.querySelector('.setup-update').textContent)`, 'update line explains updates are off');
  notes.push(`vvlDesktop descriptor: ${await js(`JSON.stringify(Object.getOwnPropertyDescriptor(window, 'vvlDesktop') ? { configurable: Object.getOwnPropertyDescriptor(window, 'vvlDesktop').configurable, writable: Object.getOwnPropertyDescriptor(window, 'vvlDesktop').writable } : null)`)}`);
  await bothThemes('01-first-run');

  // ------------------------------------------------------------ OpenRouter
  await setTheme('light');
  await choose('openrouter');
  await must(`!document.querySelector('.setup-item[data-step="engine"]') && !document.querySelector('.setup-item[data-step="model"]')`, 'OpenRouter-only shows no engine or model steps');
  // The label detector is the one download every choice needs; nothing else may mention one.
  await must(`![...document.querySelectorAll('.setup-item:not([data-step="detector"])')].some((n) => /llama|Download/.test(n.textContent))`, 'OpenRouter-only never mentions llama.cpp or downloads');
  await must(stepIs('key', 'todo'), 'key step is to do');
  await bothThemes('02-openrouter-todo');

  // Save a key without testing it (no call to OpenRouter) and add two models.
  await authFetch('PUT', '/v1/keys/openrouter', { fields: { api_key: 'sk-or-v1-0000000000000000fake' } });
  await authFetch('PUT', '/v1/settings', { openrouter: { models: ['google/gemini-3.8-flash', 'qwen/qwen3-vl-32b-instruct'] } });
  await js(`location.hash = '#vvl'; true`);
  await waitFor(`window.vvlUi.activeTab() === 'vvl'`, 'the VVL tab');
  await sleep(1700);
  await js(`document.getElementById('appVersion').click(); true`);
  await setupReady();
  await waitFor(stepIs('key', 'done') + ` && ${stepIs('pick', 'done')}`, 'key and models done');
  await must(`Boolean(document.querySelector('.setup-ready'))`, 'the ready banner for OpenRouter');
  await must(`document.getElementById('appVersion').getAttribute('aria-current') === 'page'`, 'version button marks the Setup page');
  await bothThemes('03-openrouter-ready');

  // ------------------------------------------------------------ on this computer
  await setTheme('light');
  await choose('local');
  await waitFor(`${stepIs('engine', 'todo')} && ${stepIs('model', 'todo')}`, 'engine and model steps', 30000);
  await must(`!document.querySelector('.setup-item[data-step="key"]')`, 'local-only shows no key step');
  await bothThemes('04-local-todo');

  // The recommended fake CUDA build installs, then fails its self-test.
  await setTheme('light');
  await clickIn('engine', 'Install');
  await waitFor(stepIs('engine', 'working'), 'the engine install to start', 15000);
  await waitFor(`document.querySelector('.setup-item[data-step="engine"] .job-progress')`, 'engine progress', 15000);
  await shoot('05-engine-installing-light');
  await waitFor(stepIs('engine', 'problem'), 'the CUDA self-test failure', 120000);
  await bothThemes('06-engine-problem');
  await setTheme('light');
  await clickIn('engine', 'Install');
  await waitFor(stepIs('engine', 'done'), 'the Vulkan build to install', 120000);

  // The model download. The fake catalog recommends its "bad checksum" build, so a
  // working build is started through the API instead; the page follows any download.
  await must(`/Recommended:/.test(${stepText('model')})`, 'the model step names the recommended build');
  const models = await authFetch('GET', '/v1/models');
  const small = models.local.families.flatMap((f) => f.builds).find((b) => b.id === 'demo-small');
  await authFetch('POST', '/v1/models/local/demo-small/download', { confirm_size_bytes: small.total_size_bytes });
  await js(`location.hash = '#models'; true`);
  await waitFor(`window.vvlUi.activeTab() === 'models'`, 'the Models tab');
  await sleep(1700);
  await js(`location.hash = '#setup'; true`);
  await setupReady();
  await waitFor(`${stepIs('model', 'working')} && document.querySelector('.setup-item[data-step="model"] .job-progress')`, 'model progress', 15000);
  await sleep(1500);
  await bothThemes('07-model-downloading');
  await setTheme('light');
  await waitFor(stepIs('model', 'done'), 'the model download to finish', 240000);
  await must(`Boolean(document.querySelector('.setup-ready'))`, 'the ready banner for local');
  await bothThemes('08-local-ready');

  // ------------------------------------------------------------ both
  await setTheme('light');
  await choose('both');
  await must(`Boolean(document.querySelector('.setup-item[data-step="key"]')) && Boolean(document.querySelector('.setup-item[data-step="engine"]'))`, 'both shows every step');
  await bothThemes('09-both-ready');

  // Folder change panel.
  await setTheme('light');
  await clickIn('folder', 'Change');
  await waitFor(`Boolean(document.querySelector('.setup-folder-card .mf-card'))`, 'the models folder card', 15000);
  await sleep(800);
  await js(`document.querySelector('.setup-item[data-step="folder"]').scrollIntoView({ block: 'center' }); true`);
  await shoot('10-folder-open-light');
  await clickIn('folder', 'Close');

  // ------------------------------------------------------------ update ready (fake updater)
  await js(FAKE_UPDATER);
  await sleep(300);
  await setupReady();
  await must(`/Up to date/.test(document.querySelector('.setup-update').textContent)`, 'fake updater: up to date');
  await js(`window.__fakeUpdate.set({ status: 'downloading', version: '0.2.0', percent: 45 }); true`);
  await sleep(300);
  await must(`/Downloading 0.2.0 \\(45%\\)/.test(document.querySelector('.setup-update').textContent)`, 'downloading text');
  await shoot('11-update-downloading-light');
  await js(`window.__fakeUpdate.set({ status: 'ready', version: '0.2.0', percent: 100 }); true`);
  await sleep(300);
  await must(`!document.getElementById('updateReady').hidden`, 'the update pill shows');
  await bothThemes('12-update-ready');
  await js(`document.getElementById('updateReady').click(); true`);
  await sleep(300);
  await must(`window.__fakeUpdate.installs.length === 1 && window.__fakeUpdate.installs[0].batchRunning === false`, 'install called with batchRunning false');
  // The pill on another tab.
  await js(`document.getElementById('tab-vvl').click(); true`);
  await waitFor(`window.vvlUi.activeTab() === 'vvl' && window.vvlUi.tabState('vvl') === 'ready'`, 'the VVL tab');
  await must(`typeof window.vvlUi.batchRunning === 'function' && window.vvlUi.batchRunning() === false`, 'batchRunning exposed by the VVL tab');
  await setTheme('dark');
  await shoot('13-update-pill-vvl-dark');
  await js(`window.__fakeUpdate.set({ status: 'error', error: 'The update server did not answer.' }); true`);
  await js(`document.getElementById('appVersion').click(); true`);
  await setupReady();
  await shoot('14-update-error-dark');

  // A smaller window (the shell's minimum width is about 1000 px).
  await setTheme('light');
  await resize(1000, 1000);
  await js('window.scrollTo(0, 0); true');
  await shoot('15-small-window-light');
};
