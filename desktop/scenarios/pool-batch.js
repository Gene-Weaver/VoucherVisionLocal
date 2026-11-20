// Screenshot scenario: a folder batch through the engine pool with real models.
// Chooses the OCR and parsing models from VVL_SCENARIO_OCR / VVL_SCENARIO_PARSE
// (model ids), drops four benchmark sheets into the Batch Folder input, runs the
// batch, and captures the batch while it runs, its result list, and the Engine
// tab. It never downloads, removes, installs, or uninstalls anything.
//
//   VVL_PORT=8652 VVL_HOME=... VVL_PRIVATE_PATH=... VVL_SETTINGS_PATH=... VVL_PROMPTS_USER_DIR=... \
//     npm run screenshots -- <outDir> --scenario scenarios/pool-batch.js

module.exports = async function poolBatch(t) {
  const { js, waitFor, shoot, sleep, problems, notes, gotoTab, setTheme, fs, path, REPO_ROOT } = t;
  const ocr = process.env.VVL_SCENARIO_OCR || 'local:qwen3-vl-8b-q8';
  const parse = process.env.VVL_SCENARIO_PARSE || ocr;
  const count = Number(process.env.VVL_SCENARIO_IMAGES || 4);

  await setTheme('light');
  await gotoTab('vvl');
  const selected = await js(`(() => {
    const box = (name, value) => document.querySelector('input[name="' + name + '"][value="' + value + '"]');
    const o = box('engines', ${JSON.stringify(ocr)}), p = box('llm_model', ${JSON.stringify(parse)});
    if (!o || !p || o.disabled || p.disabled) return false;
    if (!o.checked) o.click();
    for (const other of document.querySelectorAll('input[name="engines"]:checked')) if (other.value !== ${JSON.stringify(ocr)}) other.click();
    if (!p.checked) p.click();
    return true;
  })()`).catch(() => false);
  if (!selected) {
    problems.push(`could not select ${ocr} / ${parse}`);
    return;
  }
  await waitFor('window.vvlUi.inflight() === 0', 'the model choice to be saved');
  await sleep(1500);
  await js('window.scrollTo(0, 0); true');
  await shoot('pool-01-selected');

  const dir = path.join(REPO_ROOT, 'bench', 'images');
  const files = fs.readdirSync(dir).filter((f) => /\.jpe?g$/i.test(f))
    .map((f) => ({ f, size: fs.statSync(path.join(dir, f)).size })).sort((a, b) => a.size - b.size).slice(0, count);
  await js(`window.__poolFiles = []; true`);
  for (const { f } of files) {
    const b64 = fs.readFileSync(path.join(dir, f)).toString('base64');
    await js(`(() => {
      const bytes = Uint8Array.from(atob(${JSON.stringify(b64)}), (c) => c.charCodeAt(0));
      window.__poolFiles.push(new File([bytes], ${JSON.stringify(f)}, { type: 'image/jpeg' }));
      return true;
    })()`);
  }
  await js(`(() => {
    [...document.querySelectorAll('.tabs [role="tab"]')].find((x) => x.dataset.tab === 'batchFolder').click();
    const dt = new DataTransfer();
    for (const file of window.__poolFiles) dt.items.add(file);
    const input = document.getElementById('imageFolderInput');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);
  await sleep(500);
  const started = Date.now();
  await js(`document.getElementById('imageFolderInput').closest('[role="tabpanel"]').querySelector('button.button').click(); true`);
  const panel = `document.getElementById('imageFolderInput').closest('[role="tabpanel"]')`;
  const text = `((${panel}.querySelector('.progress-text') || {}).textContent || '')`;
  const seen = [];
  const shots = new Set();
  for (;;) {
    const now = await js(text).catch(() => '');
    if (now && seen[seen.length - 1] !== now) seen.push(now);
    for (const [name, pattern] of [['pool-02-reading', /Reading labels/], ['pool-03-parsing', /Parsing/], ['pool-02-processing', /Processing \d/]]) {
      if (!shots.has(name) && pattern.test(now)) {
        shots.add(name);
        await js(`document.getElementById('imageFolderInput').closest('[role="tabpanel"]').scrollIntoView({ block: 'center' }); true`);
        await shoot(name);
      }
    }
    if (/complete|Stopped/.test(now)) break;
    if (Date.now() - started > 30 * 60 * 1000) {
      problems.push('the batch did not finish in 30 minutes');
      return;
    }
    await sleep(400);
  }
  const seconds = Math.round((Date.now() - started) / 100) / 10;
  const rows = await js(`JSON.stringify([...${panel}.querySelectorAll('.result-item')].map((r) => [r.classList.contains('success'), r.textContent.slice(0, 90)]))`);
  notes.push(`batch of ${files.length} in ${seconds} s; progress texts: ${seen.join(' | ')}`);
  notes.push(`rows: ${rows}`);
  if (JSON.parse(rows).some((r) => !r[0])) problems.push('a batch row failed');
  await js(`document.getElementById('imageFolderInput').closest('[role="tabpanel"]').scrollIntoView({ block: 'center' }); true`);
  await shoot('pool-04-done');

  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await sleep(1500);
  await shoot('pool-05-engine');
  notes.push(`engine status: ${await js(`(document.querySelector('#engineSection .card') || {}).innerText || ''`)}`);
};
