// Screenshot scenario for the sticky header and tab bar, the resource monitor,
// and file names from image URLs.
//
// It never clicks Download, Remove, Install, or Uninstall. It fetches one image
// from the first example URL with the demo engine (set VVL_SCENARIO_NO_FETCH=1
// to skip that). With VVL_SCENARIO_ENGINE_MODEL=local:<build_id> it also starts
// the engine with that installed model once, to show the engine's share of
// graphics memory, and stops it again.
//
//   VVL_PORT=8652 VVL_HOME=... VVL_PRIVATE_PATH=... VVL_SETTINGS_PATH=... VVL_PROMPTS_USER_DIR=... \
//     VVL_USER_DATA_DIR=... npm run screenshots -- <outDir> --scenario scenarios/header-monitor.js

const MICH_URL = 'https://quod.lib.umich.edu/cgi/i/image/api/image/herb00ic:1122841:MICH-V-1122841/full/res:0/0/native.jpg';
const URL_LIST = [
  MICH_URL,
  'https://medialib.naturalis.nl/file/id/L.3800382/format/large',
  'https://swbiodiversity.org/imglib/h_seinet/seinet/KHD/KHD00041/KHD00041592_lg.jpg',
];

module.exports = async function headerMonitor(t) {
  const { js, waitFor, shoot, sleep, problems, notes, gotoTab, setTheme, win } = t;
  const must = async (code, what) => {
    const ok = await js(code).catch(() => false);
    if (!ok) problems.push(`check failed: ${what}`);
    return ok;
  };
  const authFetch = (method, pathName, body) => js(`fetch(${JSON.stringify(pathName)}, {method: ${JSON.stringify(method)},
    headers: {'X-VVL-Token': sessionStorage.getItem('vvl.session-token'), 'Content-Type': 'application/json'},
    body: ${body === undefined ? 'undefined' : JSON.stringify(JSON.stringify(body))}}).then((r) => r.json())`);
  // The bars are at the top of the window, opaque, and nothing of the page is in front of them.
  const barsOnTop = `(() => {
    const bar = document.getElementById('topBar').getBoundingClientRect();
    if (Math.round(bar.top) !== 0) return false;
    for (const sel of ['.header', '#tabBar']) {
      const r = document.querySelector(sel).getBoundingClientRect();
      if (r.top < 0 || r.bottom > bar.bottom + 1) return false;
      for (const x of [r.left + 12, r.left + r.width / 2, r.right - 12]) {
        const hit = document.elementFromPoint(x, r.top + r.height / 2);
        if (!hit || !hit.closest('#topBar')) return false;
      }
    }
    return true;
  })()`;

  // Give the page an exact size, whatever the window manager did with the window.
  const resize = async (width, height) => {
    if (win.isFullScreen()) win.setFullScreen(false);
    if (win.isMaximized()) win.unmaximize();
    win.setContentSize(width, height);
    const ok = await waitFor(`window.innerWidth === ${width} && window.innerHeight === ${height}`, `a ${width}x${height} page`, 5000);
    if (!ok) notes.push(`page size is ${JSON.stringify(await js('[innerWidth, innerHeight]'))}, window ${JSON.stringify(win.getBounds())}`);
    await sleep(300);
  };

  // Keyboard focus styles (the popovers open on focus) need a focused page even
  // when another window is in front of this one.
  await t.wc.debugger.sendCommand('Emulation.setFocusEmulationEnabled', { enabled: true }).catch((err) => notes.push(`focus emulation: ${err.message}`));

  const sizes = [['1100', 1100, 700], ['wide', 1800, 1000]];
  for (const [sizeName, width, height] of sizes) {
    await resize(width, height);
    for (const theme of ['light', 'dark']) {
      await setTheme(theme);
      for (const tab of ['vvl', 'models', 'engine']) {
        await gotoTab(tab);
        await waitFor(`Boolean(window.vvlUi.metrics && window.vvlUi.metrics())`, 'the first monitor reading', 15000);
        await js('window.scrollTo(0, 0); true');
        await sleep(300);
        if (await js('window.scrollY > 0')) { // late content moved the page; go back to the top
          await js('window.scrollTo(0, 0); true');
          await sleep(300);
        }
        await must(`!document.getElementById('topBar').classList.contains('is-scrolled')`, `${sizeName} ${tab} ${theme}: no shadow at the top`);
        await must(barsOnTop, `${sizeName} ${tab} ${theme}: bars at the top`);
        await must(`document.documentElement.scrollWidth <= document.documentElement.clientWidth`, `${sizeName} ${tab} ${theme}: no sideways scrolling`);
        await must(`(() => { const h = document.querySelector('.header'); return h.scrollWidth <= h.clientWidth + 1 && h.getBoundingClientRect().height < 70; })()`, `${sizeName} ${tab} ${theme}: header content fits on one line`);
        await shoot(`${sizeName}-${tab}-${theme}-top`);
        await js('window.scrollTo(0, 420); true');
        await sleep(300);
        await must('window.scrollY > 100', `${sizeName} ${tab} ${theme}: the page scrolled`);
        await must(`document.getElementById('topBar').classList.contains('is-scrolled')`, `${sizeName} ${tab} ${theme}: shadow when scrolled`);
        await must(barsOnTop, `${sizeName} ${tab} ${theme}: bars stay at the top when scrolled`);
        await shoot(`${sizeName}-${tab}-${theme}-scrolled`);
      }
    }
    const bar = await js(`(() => { const r = document.getElementById('topBar').getBoundingClientRect(); return {height: r.height, cssVar: getComputedStyle(document.documentElement).getPropertyValue('--topbar-h')}; })()`);
    notes.push(`${sizeName}: sticky bars are ${Math.round(bar.height)} px tall of ${height} px (--topbar-h ${bar.cssVar.trim()})`);
    if (sizeName === '1100' && bar.height > 130) problems.push(`the bars are ${bar.height} px tall at 1100x700`);
  }

  // ---- monitor popovers (wide window, then 1100)
  await setTheme('light');
  await gotoTab('vvl');
  await js('window.scrollTo(0, 0); true');
  const gpuCount = await js(`document.querySelectorAll('.resmon-gpu').length`);
  notes.push(`monitor shows ${gpuCount} graphics card(s); reading: ${JSON.stringify(await js('window.vvlUi.metrics()'))}`);
  for (const [name, selector] of [['cpu', '.resmon-cpu'], ['ram', '.resmon-ram'], ['gpu-first', '.resmon-gpu'], ['gpu-last', '.resmon-gpu:last-child']]) {
    if (!(await js(`Boolean(document.querySelector(${JSON.stringify(selector)}))`))) continue;
    await js(`document.querySelector(${JSON.stringify(selector)}).focus(); true`);
    await must(`(() => { const p = document.querySelector(${JSON.stringify(selector)} + ' .resmon-pop'); const r = p.getBoundingClientRect(); return getComputedStyle(p).display === 'block' && r.left >= 0 && r.right <= window.innerWidth && r.height > 40; })()`, `the ${name} popover opens inside the window`);
    await shoot(`popover-${name}-wide`);
  }
  await setTheme('dark');
  await resize(1100, 700);
  if (gpuCount) {
    await js(`document.querySelector('.resmon-gpu:last-child').focus(); true`);
    await shoot('popover-gpu-last-1100-dark');
    await js(`document.querySelector('.resmon-gpu:last-child').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true})); true`);
    await must(`getComputedStyle(document.querySelector('.resmon-gpu:last-child .resmon-pop')).display === 'none'`, 'Escape closes the popover');
    await js('document.activeElement.blur(); true');
  }
  // Numbers changing must not move anything: the monitor keeps its width across readings.
  const widthNow = () => js(`document.querySelector('.resmon').getBoundingClientRect().width`);
  const w1 = await widthNow();
  await sleep(4500);
  if (Math.abs((await widthNow()) - w1) > 0.5) problems.push('the monitor changed width between readings');
  await setTheme('light');

  // ---- in-page jumps land below the bars
  await gotoTab('models');
  const jumped = await js(`(() => { const b = [...document.querySelectorAll('#panel-models .link-button')].find((x) => /Add key|^Key$/.test(x.textContent.trim())); if (!b) return false; b.click(); return true; })()`);
  if (jumped) {
    await waitFor(`window.vvlUi.activeTab() === 'keys' && Boolean(document.querySelector('.key-card.is-highlighted'))`, 'the highlighted key card');
    await sleep(500);
    await must(`(() => { const c = document.querySelector('.key-card.is-highlighted').getBoundingClientRect(); return c.top >= document.getElementById('topBar').getBoundingClientRect().bottom; })()`, 'the key card is not hidden under the bars');
    await shoot('jump-add-a-key-1100');
  } else {
    notes.push('no "Add a key" link on the Models tab');
  }
  await js(`location.hash = '#engine/engine'; true`);
  await waitFor(`window.vvlUi.activeTab() === 'engine' && Boolean(document.getElementById('engineSection'))`, 'the engine section');
  await sleep(600);
  await must(`(() => { const top = document.getElementById('engineSection').getBoundingClientRect().top; const bar = document.getElementById('topBar').getBoundingClientRect().bottom; return top >= bar - 1; })()`, 'a jump to the engine section stops below the bars');
  await shoot('jump-engine-section-1100');
  await gotoTab('vvl');
  await resize(1400, 860);

  // ---- the planned part follows the model choice at once
  const before = await js(`(window.vvlUi.metrics().gpus || []).reduce((n, g) => n + (g.planned_bytes || 0), 0)`);
  notes.push(`planned bytes with the saved selection: ${before}`);
  await gotoTab('vvl');
  await js(`(() => {
    for (const name of ['engines', 'llm_model']) { const d = document.querySelector('input[name="' + name + '"][value="demo:mock"]'); if (d && !d.checked) d.click(); }
    for (const box of document.querySelectorAll('input[name="engines"]:checked')) if (box.value !== 'demo:mock') box.click();
    return true;
  })()`);
  // The monitor asks again as soon as the choice is saved, not at its next two-second reading.
  const clickedAt = Date.now();
  const cleared = await waitFor(`(window.vvlUi.metrics().gpus || []).every((g) => !g.planned_bytes) && window.vvlUi.metrics().plan.mode === 'none'`, 'the planned part to clear after choosing the demo engine', 8000);
  const timing = await js(`(() => {
    const entries = performance.getEntriesByType('resource');
    const saves = entries.filter((e) => e.name.endsWith('/v1/selection'));
    const lastSave = saves[saves.length - 1];
    const next = lastSave ? entries.find((e) => e.name.endsWith('/v1/metrics') && e.startTime >= lastSave.responseEnd) : null;
    return { saves: saves.slice(-3).map((e) => Math.round(e.duration)), metricsAfterSaveMs: next ? Math.round(next.startTime - lastSave.responseEnd) : null, visible: document.visibilityState };
  })()`);
  notes.push(`planned part cleared: ${cleared} after ${Date.now() - clickedAt} ms; ${JSON.stringify(timing)}`);
  if (before > 0 && cleared && (timing.metricsAfterSaveMs === null || timing.metricsAfterSaveMs > 250)) {
    problems.push(`the monitor did not ask again right after the model choice was saved (${timing.metricsAfterSaveMs} ms)`);
  }
  await js('window.scrollTo(0, 0); true');
  await shoot('monitor-demo-selected');

  // ---- file names from URLs
  const names = await authFetch('POST', '/v1/url-names', { urls: URL_LIST });
  notes.push(`/v1/url-names: ${JSON.stringify(names)}`);
  await js(`(() => {
    const dt = new DataTransfer();
    dt.items.add(new File([${JSON.stringify(URL_LIST.join('\n'))}], 'urls.txt', { type: 'text/plain' }));
    const input = document.getElementById('batchUrlFileInput');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    [...document.querySelectorAll('.tabs [role="tab"]')].find((x) => x.dataset.tab === 'batchUrls').click();
    return true;
  })()`);
  await waitFor(`/MICH-V-1122841\\.jpg/.test(document.querySelector('#batchUrlFileInput').closest('.form-group').querySelector('.file-list').textContent)`, 'the parser name in the URL list preview', 5000);
  await must(`(() => { const t = document.querySelector('#batchUrlFileInput').closest('.form-group').querySelector('.file-list').textContent; return t.includes('L.3800382.jpg') && t.includes('KHD00041592_lg.jpg') && !/native\\s—/.test(t); })()`, 'every URL in the list preview has its parser name');
  await js(`document.getElementById('batchUrlFileInput').scrollIntoView({block: 'center'}); true`);
  await shoot('url-names-batch-list');

  if (!process.env.VVL_SCENARIO_NO_FETCH) {
    await js(`(() => {
      window.__downloads = [];
      const click = HTMLAnchorElement.prototype.click;
      HTMLAnchorElement.prototype.click = function () { if (this.download) { window.__downloads.push(this.download); return; } return click.call(this); };
      [...document.querySelectorAll('.tabs [role="tab"]')].find((x) => x.dataset.tab === 'url').click();
      const input = document.getElementById('imageUrl');
      input.value = ${JSON.stringify(MICH_URL)};
      input.closest('[role="tabpanel"]').querySelector('button.button').click();
      return true;
    })()`);
    await waitFor(`Boolean(document.querySelector('.vv-loader-title') || document.querySelector('.split-results-container'))`, 'the run to begin', 10000);
    const title = await js(`(document.querySelector('.vv-loader-title') || {}).textContent || ''`);
    notes.push(`loader title: ${title}`);
    if (title && title !== 'Processing MICH-V-1122841.jpg') problems.push(`the run is labeled "${title}"`);
    if (title) await shoot('url-names-single-processing');
    const finished = await waitFor(`Boolean(document.querySelector('.split-results-container') || document.querySelector('.results .state-error'))`, 'the URL result', 120000);
    if (finished && await js(`Boolean(document.querySelector('.results .state-error'))`)) {
      notes.push(`the URL run did not finish (remote server?): ${await js(`document.querySelector('.results .state-error').textContent`)}`);
    } else if (finished) {
      const filename = await js(`(() => { const m = /"filename": "([^"]*)"/.exec(document.querySelector('.json-content').textContent); return m ? m[1] : null; })()`);
      notes.push(`filename in the response: ${filename}`);
      if (filename !== 'MICH-V-1122841.jpg') problems.push(`the response filename is ${filename}`);
      await js(`[...document.querySelectorAll('.json-controls button')].find((b) => /^JSON$/.test(b.textContent.trim())).click(); true`);
      const downloads = await js('window.__downloads');
      notes.push(`download names: ${JSON.stringify(downloads)}`);
      if (downloads[0] !== 'MICH-V-1122841.json') problems.push(`the JSON download is named ${downloads[0]}`);
      await js(`document.querySelector('.json-controls').scrollIntoView({block: 'center'}); true`);
      await sleep(1500);
      await shoot('url-names-single-result');
    }
  }

  // ---- optional: the engine's share of graphics memory, with a real model
  const model = process.env.VVL_SCENARIO_ENGINE_MODEL;
  if (model) {
    await authFetch('PUT', '/v1/selection', { ocr: [model], parse: model });
    await gotoTab('engine');
    await gotoTab('vvl');
    await js('window.scrollTo(0, 0); true');
    await waitFor(`(window.vvlUi.metrics().gpus || []).some((g) => g.planned_bytes > 0)`, 'the planned part for the selected model', 8000);
    await js(`(() => { const g = [...document.querySelectorAll('.resmon-gpu')].find((n) => parseFloat(n.querySelector('.resmon-seg-planned').style.width) > 0); if (g) g.focus(); return true; })()`);
    notes.push(`before start: ${JSON.stringify(await js('window.vvlUi.metrics()'))}`);
    await shoot('engine-01-planned');
    await js('document.activeElement.blur(); true');
    const started = await authFetch('POST', '/v1/engine/start', { model_id: model });
    notes.push(`engine start: state ${started && started.managed && started.managed.state}`);
    await waitFor(`['loading_model', 'ready', 'error'].includes(window.vvlUi.metrics().engine.state)`, 'the engine to begin loading', 60000);
    await sleep(6000);
    await js(`(() => { const g = [...document.querySelectorAll('.resmon-gpu')].find((n) => parseFloat(n.querySelector('.resmon-seg-engine').style.width) > 0); if (g) g.focus(); return true; })()`);
    notes.push(`while loading: ${JSON.stringify(await js('window.vvlUi.metrics()'))}`);
    await shoot('engine-02-loading');
    const ready = await waitFor(`['ready', 'error'].includes(window.vvlUi.metrics().engine.state)`, 'the engine to be ready', 600000);
    await sleep(3000);
    const loaded = await js('window.vvlUi.metrics()');
    notes.push(`loaded: ${JSON.stringify(loaded)}`);
    if (ready && loaded.engine.state === 'ready') {
      const card = loaded.gpus.find((g) => g.engine_bytes > 0);
      if (!card) problems.push('no card shows engine memory while the model is loaded');
      if (loaded.gpus.some((g) => g.planned_bytes > 0)) problems.push('a loaded model is still counted as planned');
    } else {
      problems.push(`the engine did not become ready: ${JSON.stringify(loaded.engine)}`);
    }
    await shoot('engine-03-loaded');
    await js('document.activeElement.blur(); true');
    await authFetch('POST', '/v1/engine/stop');
    await waitFor(`window.vvlUi.metrics().engine.state === 'stopped' && window.vvlUi.metrics().gpus.every((g) => !g.engine_bytes)`, 'the engine memory to be released', 60000);
    notes.push(`after stop: ${JSON.stringify(await js('window.vvlUi.metrics()'))}`);
    await shoot('engine-04-stopped');
  }
};
