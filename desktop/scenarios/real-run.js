// Screenshot scenario for one real run with an installed local model.
// It only looks at the Models and Engine tabs and never clicks Download,
// Remove, Install, or Uninstall. Choose the model with VVL_SCENARIO_MODEL
// (default local:qwen3-vl-32b-q8). The engine is left running at the end so
// the shell's quit path stops it.
//
//   VVL_PORT=8652 VVL_HOME=... VVL_PRIVATE_PATH=... VVL_SETTINGS_PATH=... VVL_PROMPTS_USER_DIR=... \
//     npm run screenshots -- <outDir> --scenario scenarios/real-run.js

module.exports = async function realRun(t) {
  const { js, waitFor, shoot, sleep, problems, notes, gotoTab, setTheme, fs, path, REPO_ROOT } = t;
  const model = process.env.VVL_SCENARIO_MODEL || 'local:qwen3-vl-32b-q8';
  const must = async (code, what) => {
    const ok = await js(code).catch(() => false);
    if (!ok) problems.push(`check failed: ${what}`);
    return ok;
  };

  await setTheme('light');
  await gotoTab('models');
  await js('window.scrollTo(0, 0); true');
  await shoot('real-01-models');
  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('real-02-engine-stopped');

  await gotoTab('vvl');
  const selected = await must(`(() => {
    const pick = (name) => { const n = document.querySelector('input[name="' + name + '"][value=${JSON.stringify(model)}]'); if (!n || n.disabled) return false; if (!n.checked) n.click(); return true; };
    if (!pick('engines') || !pick('llm_model')) return false;
    for (const box of document.querySelectorAll('input[name="engines"]:checked')) if (box.value !== ${JSON.stringify(model)}) box.click();
    return true;
  })()`, `${model} can be selected`);
  if (!selected) return;
  await waitFor('window.vvlUi.inflight() === 0', 'the model choice to be saved');
  await js('window.scrollTo(0, 0); true');
  await shoot('real-03-vvl-selected');

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
    return true;
  })()`);

  const stageNow = `((document.querySelector('.vv-loader-stage') || {}).textContent || '')`;
  const finished = `Boolean(document.querySelector('.split-results-container') || document.querySelector('.results .state-error'))`;

  async function run(tag, shots) {
    const seen = [];
    const started = Date.now();
    await js(`document.getElementById('fileInput').closest('[role="tabpanel"]').querySelector('button.button').click(); true`);
    const taken = new Set();
    for (;;) {
      if (await js(finished).catch(() => false)) break;
      if (Date.now() - started > 20 * 60 * 1000) {
        problems.push(`${tag}: no result after 20 minutes`);
        return;
      }
      const stage = await js(stageNow).catch(() => '');
      if (stage && seen[seen.length - 1]?.stage !== stage) seen.push({ stage, at: Math.round((Date.now() - started) / 100) / 10 });
      for (const [name, pattern] of shots) {
        if (!taken.has(name) && pattern.test(stage)) {
          taken.add(name);
          await js(`document.querySelector('.vv-loader').scrollIntoView({ block: 'center' }); true`);
          await shoot(name);
        }
      }
      await sleep(300);
    }
    const seconds = Math.round((Date.now() - started) / 100) / 10;
    const failed = await js(`Boolean(document.querySelector('.results .state-error'))`);
    notes.push(`${tag}: ${failed ? 'FAILED' : 'ok'} in ${seconds} s; stages: ${seen.map((s) => `${s.at}s ${s.stage}`).join(' | ')}`);
    if (failed) problems.push(`${tag} failed: ${await js(`document.querySelector('.results .state-error').textContent`)}`);
    return !failed;
  }

  // First run: the engine starts and loads the model.
  const ok = await run('first run (cold)', [['real-04-run-loading', /Loading|Starting/], ['real-05-run-reading', /Reading/]]);
  if (ok) {
    await must(`!document.querySelector('.results .notice-demo')`, 'a real result has no demo notice');
    await sleep(2500);
    await js(`document.querySelector('.step-container:last-of-type').scrollIntoView(); true`);
    await shoot('real-06-result');
    notes.push(`first result: ${await js(`JSON.stringify(Object.fromEntries([...document.querySelectorAll('.field-table tr')].slice(0, 12).map((r) => [r.children[0].textContent, r.children[1].textContent])))`)}`);
    await js(`[...document.querySelectorAll('.result-tab')].find((b) => b.textContent === 'OCR').click(); true`);
    await shoot('real-07-result-ocr');
    await js(`[...document.querySelectorAll('.result-tab')].find((b) => b.textContent === 'Usage').click(); true`);
    await shoot('real-08-result-usage');
    // Second run: the model is already loaded.
    await run('second run (warm)', [['real-09-run-warm', /Reading/]]);
  }

  await gotoTab('engine');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('real-10-engine-ready');
  await js(`document.querySelector('.logs[role="log"]').scrollIntoView({ block: 'center' }); true`);
  await shoot('real-11-engine-log');
  await setTheme('dark');
  await js(`document.getElementById('engineSection').scrollIntoView(); true`);
  await shoot('real-12-engine-ready-dark');
  await gotoTab('vvl');
  await js(`document.querySelector('.step-container:last-of-type').scrollIntoView(); true`);
  await shoot('real-13-result-dark');
  await setTheme('light');
};
