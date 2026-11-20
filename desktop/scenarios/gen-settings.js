// Screenshot scenario for the model settings row under the OCR and Parsing model
// lists (ui/js/tabs/gen_settings.js): reasoning, strength, long output, high
// detail, context, and the values sent.
//
// It changes only saved settings (selection and generation, through the API and
// the chips); it never starts an engine. Use scratch settings and a models_dir
// that holds the models named below.
//
//   VVL_PORT=8656 VVL_HOME=... VVL_SETTINGS_PATH=... VVL_PRIVATE_PATH=... VVL_PROMPTS_USER_DIR=... \
//     VVL_USER_DATA_DIR=... npm run screenshots -- <outDir> --scenario scenarios/gen-settings.js

const QWEN35 = 'local:qwen3.5-122b-a10b-q4';
const MUSE = 'local:muse-glimmer-30b-q8';
const QWEN3VL = 'local:qwen3-vl-32b-q8';

module.exports = async function genSettings(t) {
  const { js, waitFor, shoot, sleep, problems, notes, gotoTab, setTheme, win, wc } = t;
  const authFetch = (method, pathName, body) => js(`fetch(${JSON.stringify(pathName)}, {method: ${JSON.stringify(method)},
    headers: {'X-VVL-Token': sessionStorage.getItem('vvl.session-token'), 'Content-Type': 'application/json'},
    body: ${body === undefined ? 'undefined' : JSON.stringify(JSON.stringify(body))}}).then((r) => r.json())`);
  const must = async (code, what) => {
    const ok = await js(code).catch(() => false);
    if (!ok) problems.push(`check failed: ${what}`);
    return ok;
  };
  const row = (step) => `document.querySelector('.settings-card .gen-row[aria-label^="${step === 'ocr' ? 'OCR' : 'Parsing'}"]')`;
  // Every control in the row sits on one line and inside the card.
  const oneLine = (step) => `(() => {
    const r = ${row(step)};
    if (!r || r.hidden) return false;
    const box = r.getBoundingClientRect();
    const items = [...r.children].map((n) => n.getBoundingClientRect());
    const mid = (i) => (i.top + i.bottom) / 2;
    return items.every((i) => Math.abs(mid(i) - mid(items[0])) <= 3 && i.right <= box.right + 1);
  })()`;
  const click = (selector) => js(`(() => { const n = document.querySelector(${JSON.stringify(selector)}); if (!n) return false; n.click(); return true; })()`);
  const settle = () => waitFor('window.vvlUi.inflight() === 0', 'saves to finish', 10000);

  await wc.debugger.sendCommand('Emulation.setFocusEmulationEnabled', { enabled: true }).catch((err) => notes.push(`focus emulation: ${err.message}`));
  if (win.isMaximized()) win.unmaximize();
  win.setContentSize(1280, 900);
  await waitFor('window.innerWidth === 1280', 'a 1280 px page', 5000);

  async function setup(selection, generation) {
    await authFetch('PUT', '/v1/settings', { selection });
    await authFetch('PUT', '/v1/generation', { context: { [QWEN35]: 'auto', [MUSE]: 'auto', [QWEN3VL]: 'auto' }, ...generation });
    await js('location.reload(); true').catch(() => {});
    await sleep(800);
    await waitFor(`Boolean(window.vvlUi && document.getElementById('tab-vvl'))`, 'the app shell after reload');
    await gotoTab('vvl');
    await waitFor(`Boolean(${row('ocr')} && !${row('ocr')}.hidden && ${row('parse')} && !${row('parse')}.hidden)`, 'both settings rows');
    await js(`${row('ocr')}.scrollIntoView({block: 'center'}); true`);
    await sleep(200);
  }
  const offAll = { ocr: { reasoning: 'off', strength: 'high', output_length: 'standard', image_detail: 'standard' },
    parse: { reasoning: 'off', strength: 'high', output_length: 'standard' } };

  // 1. Qwen3.5 for both steps: reasoning on/off, no strength levels.
  await setup({ ocr: [QWEN35], parse: QWEN35 }, offAll);
  await must(oneLine('ocr'), 'Qwen3.5 OCR row is one line');
  await must(`!document.querySelector('[data-fk="ocr-strength"]')`, 'no strength menu for Qwen3.5');
  await shoot('gen-1-qwen35');
  // A chip click saves and keeps keyboard focus on the chip.
  await js(`document.querySelector('[data-fk="ocr-output_length"]').focus(); document.querySelector('[data-fk="ocr-output_length"]').click(); true`);
  await settle();
  await must(`document.querySelector('[data-fk="ocr-output_length"]').getAttribute('aria-pressed') === 'true'`, 'Long output is on after a click');
  await must(`document.activeElement && document.activeElement.dataset.fk === 'ocr-output_length'`, 'focus stays on the chip after saving');
  const gen1 = await authFetch('GET', '/v1/generation');
  if (!gen1 || gen1.ocr.values.output_length !== 'long' || gen1.ocr.models[0].sent.max_tokens !== 8192) {
    problems.push(`long output not saved: ${JSON.stringify(gen1 && gen1.ocr.values)} ${gen1 && gen1.ocr.models[0].sent.max_tokens}`);
  }
  // Hover the reasoning chip to show its hint.
  const rect = await js(`(() => { const r = document.querySelector('[data-fk="ocr-reasoning"]').getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()`);
  wc.sendInputEvent({ type: 'mouseMove', x: Math.round(rect.x), y: Math.round(rect.y) });
  await sleep(900);
  await shoot('gen-2-hint');
  wc.sendInputEvent({ type: 'mouseMove', x: 5, y: 5 });

  // 2. Muse with reasoning on: the strength menu.
  await setup({ ocr: [MUSE], parse: MUSE }, { ocr: { ...offAll.ocr, reasoning: 'on' }, parse: { ...offAll.parse, reasoning: 'on' } });
  await must(oneLine('ocr'), 'Muse OCR row is one line');
  await click('[data-fk="ocr-strength"]');
  await sleep(200);
  await must(`document.querySelectorAll('.gen-menu-item').length === 4`, 'four strength levels');
  await shoot('gen-3-muse-strength');
  await click('[data-fk="ocr-level-medium"]');
  await settle();
  const gen2 = await authFetch('GET', '/v1/generation');
  const museSent = gen2 && gen2.ocr.models[0].sent;
  if (!museSent || JSON.stringify(museSent.chat_template_kwargs) !== JSON.stringify({ reasoning_strength: 'medium' })) {
    problems.push(`strength not sent: ${JSON.stringify(museSent)}`);
  }
  await must(`!document.querySelector('.gen-menu')`, 'the menu closes after a pick');
  // Reasoning off for Muse: it still gets the reasoning token limit.
  await click('[data-fk="parse-reasoning"]');
  await settle();
  const gen3 = await authFetch('GET', '/v1/generation');
  if (!gen3 || gen3.parse.models[0].sent.max_tokens !== 8192 || gen3.parse.models[0].sent.chat_template_kwargs.reasoning_strength !== 'low') {
    problems.push(`Muse parse with reasoning off: ${JSON.stringify(gen3 && gen3.parse.models[0].sent)}`);
  }

  // 3. Two OCR models (one cannot reason) and Qwen3.5 parsing; the context menu.
  await setup({ ocr: [QWEN3VL, MUSE], parse: QWEN35 }, { ocr: { ...offAll.ocr, reasoning: 'on' }, parse: offAll.parse });
  await must(oneLine('ocr'), 'two-model OCR row is one line');
  await click('[data-fk="ocr-context"]');
  await sleep(200);
  await must(`document.querySelectorAll('.gen-pop .gen-ctx-input').length === 2`, 'one context line per OCR model');
  await shoot('gen-4-mixed-context');
  await js(`document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true})); true`);
  await sleep(100);
  await must(`!document.querySelector('.gen-pop')`, 'Escape closes the menu');
  await click('[data-fk="parse-raw"]');
  await sleep(200);
  await must(`document.querySelector('.gen-raw-pre') && document.querySelector('.gen-raw-pre').textContent.includes('enable_thinking')`, 'values sent are shown');
  await shoot('gen-5-values-sent');
  await click('[data-fk="parse-raw"]');

  // 4. Dark theme.
  await setTheme('dark');
  await sleep(300);
  await shoot('gen-6-dark');
  await setTheme('light');
};
