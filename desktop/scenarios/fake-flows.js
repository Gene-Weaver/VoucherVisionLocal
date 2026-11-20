// Screenshot scenario for the download / install / engine flows.
// Run it against tools/dev_fake_backend.py (a fresh --root each time), never
// against a real models folder:
//
//   uv run python tools/dev_fake_backend.py --root <scratch> --rate 1.5
//   VVL_UI_URL='http://127.0.0.1:8653/?token=dev-fake-token' \
//     npm run screenshots -- <outDir> --scenario scenarios/fake-flows.js

// Small page-side helpers: find a build row by its name, click its buttons, drive dialogs.
const HELPERS = `window.__t = {
  row(label) { return [...document.querySelectorAll('.tab-panel:not([hidden]) tr')].find((r) => { const s = r.querySelector('.build-name strong'); return s && s.textContent === label; }) || null; },
  rowText(label) { const r = this.row(label); if (!r) return ''; const d = r.nextElementSibling && r.nextElementSibling.classList.contains('build-detail-row') && !r.nextElementSibling.hidden ? r.nextElementSibling.textContent : ''; return r.textContent + ' || ' + d; },
  button(label, text) { const r = this.row(label); const rows = r ? [r, r.nextElementSibling].filter(Boolean) : []; for (const x of rows) { const b = [...x.querySelectorAll('button')].find((n) => n.textContent.trim().startsWith(text)); if (b) return b; } return null; },
  click(label, text) { const b = this.button(label, text); if (!b || b.disabled) return false; b.click(); return true; },
  dialog() { return document.querySelector('.modal'); },
  confirm() { const b = document.querySelector('.modal .modal-actions .button'); if (!b || b.disabled) return false; b.click(); return true; },
  closeDialog() { const b = document.querySelector('.modal .modal-close'); if (b) b.click(); return true; },
  byText(sel, text) { return [...document.querySelectorAll(sel)].find((n) => n.textContent.trim().startsWith(text)) || null; },
}; true`;

module.exports = async function fakeFlows(t) {
  const { js, waitFor, shoot, sleep, problems, notes, gotoTab, setTheme, expectHttp, fs, path, REPO_ROOT } = t;

  await js(HELPERS);

  const must = async (code, what) => {
    const ok = await js(code).catch(() => false);
    if (!ok) problems.push(`check failed: ${what}`);
    return ok;
  };
  const rowHas = (label, text, timeoutMs = 60000) => waitFor(`window.__t.rowText(${JSON.stringify(label)}).includes(${JSON.stringify(text)})`, `"${text}" in the ${label} row`, timeoutMs);
  const click = (label, text) => must(`window.__t.click(${JSON.stringify(label)}, ${JSON.stringify(text)})`, `click ${text} on ${label}`);
  const freeSpaceShown = () => waitFor(`Boolean(window.__t.dialog()) && /free at/.test(window.__t.dialog().textContent)`, 'free disk space in the dialog', 60000);
  const dialogOpen = () => waitFor('Boolean(window.__t.dialog())', 'a dialog', 10000);
  const dialogGone = () => waitFor('!window.__t.dialog()', 'the dialog to close', 20000);
  const confirm = () => must('window.__t.confirm()', 'confirm the dialog');

  const SMALL = 'Demo-VL Small (Q8)';
  const MEDIUM = 'Demo-VL Medium (Q8)';
  const BAD = 'Demo-VL Bad Checksum (Q8)';
  const HUGE = 'Demo-VL Huge (Q8)';
  const CUDA = 'Fake build, NVIDIA CUDA 12';
  const VULKAN = 'Fake build, Vulkan';

  // ------------------------------------------------------------ downloads
  await setTheme('light');
  await gotoTab('models');
  await must(`document.querySelector('#panel-models').textContent.includes('not installed yet')`, 'the engine-not-installed notice');
  await shoot('01-models-start');

  await click(MEDIUM, 'Download');
  await dialogOpen();
  await freeSpaceShown();
  await shoot('02-download-dialog');
  await confirm();
  await dialogGone();
  await rowHas(MEDIUM, 'left');
  await must(`!window.__t.button(${JSON.stringify(SMALL)}, 'Download').disabled`, 'other Download buttons stay usable during a download');
  await must(`!document.querySelector('.job-indicator').hidden`, 'the header indicator shows');
  await must(`document.querySelector('#panel-models [role="progressbar"]').getAttribute('aria-valuenow') !== null`, 'progress bar has ARIA values');
  await shoot('03-downloading');

  // A second download waits in line, and can be taken out of the line.
  await click(SMALL, 'Download');
  await dialogOpen();
  await must(`/Add to the download queue/.test(window.__t.dialog().textContent)`, 'the dialog offers to queue the download');
  await shoot('03a-queue-dialog');
  await confirm();
  await dialogGone();
  await rowHas(SMALL, 'Queued');
  await rowHas(SMALL, 'Next in line');
  await waitFor(`/1 more queued/.test(document.querySelector('.job-indicator').textContent)`, 'the header indicator to count the queued download', 10000);
  await shoot('03b-queued');
  await click(SMALL, 'Cancel');
  await rowHas(SMALL, 'Not installed');

  // Progress survives a tab switch and a page reload.
  await gotoTab('vvl');
  await shoot('04-indicator-on-vvl-tab');
  await js(`document.querySelector('.job-indicator').click(); true`);
  await waitFor(`window.vvlUi.activeTab() === 'models'`, 'the indicator to open the Models tab');
  await js('window.location.reload(); true');
  await sleep(1500);
  await waitFor(`Boolean(window.vvlUi && document.getElementById('tab-models'))`, 'the page after a reload');
  await js(HELPERS); // the helpers were lost with the reload
  await gotoTab('models');
  await rowHas(MEDIUM, 'left');
  await shoot('05-downloading-after-reload');

  // Cancel, then resume.
  await click(MEDIUM, 'Cancel');
  await rowHas(MEDIUM, 'Paused');
  await shoot('06-paused');
  await click(MEDIUM, 'Resume download');
  await dialogOpen();
  await freeSpaceShown();
  await must(`Boolean(window.__t.dialog()) && window.__t.dialog().textContent.includes('Already downloaded')`, 'the resume dialog says what is already downloaded');
  await shoot('07-resume-dialog');
  await confirm();
  await dialogGone();
  await rowHas(MEDIUM, 'on disk', 120000);
  await shoot('08-installed');

  // A file that fails its checksum.
  await click(BAD, 'Download');
  await dialogOpen();
  await confirm();
  await dialogGone();
  await rowHas(BAD, 'Download failed', 120000);
  await shoot('09-checksum-failure');

  // A model that does not fit on the disk.
  await click(HUGE, 'Download');
  await dialogOpen();
  await freeSpaceShown();
  expectHttp(507, '/download');
  await confirm();
  await waitFor(`Boolean(document.querySelector('.modal .modal-problem:not([hidden])'))`, 'the disk space message');
  await shoot('10-not-enough-disk');
  await js('window.__t.closeDialog()');

  // The small model, for the engine.
  await click(SMALL, 'Download');
  await dialogOpen();
  await confirm();
  await dialogGone();
  await rowHas(SMALL, 'on disk', 120000);

  // ------------------------------------------------------------ engine install
  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('11-engine-not-installed');
  await click(CUDA, 'Install');
  await dialogOpen();
  await shoot('12-install-dialog');
  await confirm();
  await dialogGone();
  await rowHas(CUDA, 'left');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('13-installing');
  await rowHas(CUDA, 'cannot run on this computer', 120000);
  await must(`Boolean(window.__t.button(${JSON.stringify(CUDA)}, 'Install ${VULKAN} instead'))`, 'the next build is offered');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('14-self-test-failed');
  await click(CUDA, `Install ${VULKAN} instead`);
  await dialogOpen();
  await confirm();
  await dialogGone();
  await rowHas(VULKAN, 'Uninstall', 120000);
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('15-engine-installed');

  // ------------------------------------------------------------ engine settings
  await js(`(() => {
    const set = (id, v) => { const n = document.getElementById(id); n.value = v; n.dispatchEvent(new Event('input', { bubbles: true })); n.dispatchEvent(new Event('change', { bubbles: true })); };
    set('engParallel', '3');
    set('engIdle', '20');
    set('engCtxMode', 'custom');
    set('engCtx', '8192');
    document.getElementById('engGpuPick').click();
    const boxes = document.querySelectorAll('.gpu-choices input');
    if (boxes.length) boxes[boxes.length - 1].click();
    return true;
  })()`);
  await js(`window.__t.byText('#panel-engine button', 'Save settings').click(); true`);
  await waitFor(`[...document.querySelectorAll('.toast')].some((n) => n.textContent.includes('Engine settings saved'))`, 'the settings to save');
  await js(`window.__t.byText('#panel-engine button', 'Save settings').scrollIntoView({ block: 'end' }); true`);
  await shoot('16-engine-settings-saved');

  // ------------------------------------------------------------ start, log, run
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await must(`document.getElementById('engineModel').options.length === 2`, 'both installed models are offered for Start');
  await js(`(() => { const s = document.getElementById('engineModel'); s.value = 'local:demo-small'; s.dispatchEvent(new Event('change', { bubbles: true })); document.getElementById('engineStart').click(); return true; })()`);
  await waitFor(`document.querySelector('.engine-status').textContent.includes('Loading model') || document.querySelector('.engine-status').textContent.includes('Starting')`, 'the engine to begin starting');
  await shoot('17-engine-loading');
  await waitFor(`document.querySelector('.engine-status .badge').textContent === 'Ready'`, 'the engine to be ready', 60000);
  await waitFor(`document.querySelectorAll('.logs[role="log"] .log-line').length > 0`, 'engine log lines');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('18-engine-ready');
  await js(`document.querySelector('.logs[role="log"]').scrollIntoView({ block: 'center' }); true`);
  await shoot('19-engine-log');

  // Removing the loaded model is refused.
  await gotoTab('models');
  await click(SMALL, 'Remove');
  await dialogOpen();
  expectHttp(409, '/v1/models/local/demo-small');
  await confirm();
  await waitFor(`Boolean(document.querySelector('.modal .modal-problem:not([hidden])'))`, 'the model-in-use message');
  await shoot('20-remove-while-loaded');
  await js('window.__t.closeDialog()');

  // A real request through the fake engine, from the VVL tab.
  await gotoTab('vvl');
  await must(`(() => {
    const pick = (name, value) => { const n = document.querySelector('input[name="' + name + '"][value="' + value + '"]'); if (!n || n.disabled) return false; if (!n.checked) n.click(); return true; };
    if (!pick('engines', 'local:demo-small') || !pick('llm_model', 'local:demo-small')) return false;
    for (const box of document.querySelectorAll('input[name="engines"]:checked')) if (box.value !== 'local:demo-small') box.click();
    return true;
  })()`, 'the installed local model can be selected');
  await waitFor('window.vvlUi.inflight() === 0', 'the model choice to be saved');
  await must(`[...document.querySelectorAll('input[name="engines"]:checked')].map((n) => n.value).join() === 'local:demo-small' && document.querySelector('input[name="llm_model"]:checked').value === 'local:demo-small'`, 'the local model is the only one selected');
  await must(`document.querySelector('input[name="engines"][value="local:demo-huge"]').disabled`, 'a model that is not installed stays disabled');
  await js('window.scrollTo(0, 0); true');
  await shoot('21-vvl-local-selected');
  const sample = path.join(REPO_ROOT, 'tests', 'fixtures', 'images', 'MICH_16205594_Poaceae_Jouvea_pilosa.jpg');
  const b64 = fs.readFileSync(sample).toString('base64');
  await js(`(async () => {
    const bytes = Uint8Array.from(atob(${JSON.stringify(b64)}), (c) => c.charCodeAt(0));
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], 'MICH_16205594_Poaceae_Jouvea_pilosa.jpg', { type: 'image/jpeg' }));
    const input = document.getElementById('fileInput');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    [...document.querySelectorAll('.tabs [role="tab"]')].find((x) => x.dataset.tab === 'file').click();
    input.closest('[role="tabpanel"]').querySelector('button.button').click();
    return true;
  })()`);
  await waitFor(`/Reading the label/.test((document.querySelector('.vv-loader-stage') || {}).textContent || '')`, 'the reading-the-label state', 60000);
  await sleep(1200);
  await must(`[...document.querySelectorAll('#panel-vvl .tab-content button.button')].every((b) => b.disabled)`, 'Run is disabled while a request runs');
  await js(`document.querySelector('.vv-loader').scrollIntoView({ block: 'center' }); true`);
  await shoot('22-vvl-running');
  await waitFor(`Boolean(document.querySelector('.split-results-container') || document.querySelector('.results .state-error'))`, 'a result', 180000);
  await must(`Boolean(document.querySelector('.split-results-container')) && !document.querySelector('.results .notice-demo')`, 'a local result without the demo notice');
  await js(`document.querySelector('.step-container:last-of-type').scrollIntoView(); true`);
  await shoot('23-vvl-result');

  // Two different local models: the warning, and the batch limit note.
  await js(`document.querySelector('input[name="llm_model"][value="local:demo-medium"]').click(); true`);
  await waitFor(`Boolean(document.querySelector('.model-warning .notice'))`, 'the model-switching warning');
  await js('window.scrollTo(0, 0); true');
  await shoot('24-vvl-two-models-warning');
  await js(`document.querySelector('input[name="llm_model"][value="local:demo-small"]').click(); [...document.querySelectorAll('.tabs [role="tab"]')].find((x) => x.dataset.tab === 'batchFolder').click(); true`);
  await must(`document.getElementById('batchFolderConcurrency').value === '3' && !document.getElementById('batchFolderConcurrencyNote').hidden`, 'batch concurrency follows the engine setting (3)');
  await js(`document.getElementById('batchFolderConcurrency').scrollIntoView({ block: 'center' }); true`);
  await shoot('25-vvl-batch-limit');
  // Leave the demo engine selected, as a later tour expects.
  await js(`(() => { document.querySelector('input[name="llm_model"][value="demo:mock"]').click(); document.querySelector('input[name="engines"][value="demo:mock"]').click(); document.querySelector('input[name="engines"][value="local:demo-small"]').click(); return true; })()`);

  // ------------------------------------------------------------ dark theme, stop, remove, uninstall
  await setTheme('dark');
  await gotoTab('models');
  await js('window.scrollTo(0, 0); true');
  await shoot('26-models-dark');
  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('27-engine-ready-dark');
  await js(`document.getElementById('engineStop').click(); true`);
  await waitFor(`document.querySelector('.engine-status .badge').textContent === 'Stopped'`, 'the engine to stop', 60000);
  await shoot('28-engine-stopped-dark');

  await gotoTab('models');
  await click(MEDIUM, 'Remove');
  await dialogOpen();
  await shoot('29-remove-dialog-dark');
  await confirm();
  await dialogGone();
  await rowHas(MEDIUM, 'Not installed');

  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await click(VULKAN, 'Uninstall');
  await dialogOpen();
  await shoot('30-uninstall-dialog-dark');
  await confirm();
  await dialogGone();
  await waitFor(`document.querySelector('.engine-status .badge').textContent === 'Not installed'`, 'the engine to be uninstalled');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('31-engine-uninstalled-dark');
  await setTheme('light');
};
