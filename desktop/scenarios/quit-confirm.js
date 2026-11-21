// Scenario: quitting while a download runs asks first. Fake backend only
// (tools/dev_fake_backend.py); it starts and then cancels a download.
// With VVL_SCENARIO_NATIVE_DIALOG=1 the real dialog is opened and left on
// screen (for a screen capture); the run then has to be stopped from outside.

const { dialog } = require('electron');

module.exports = async function quitConfirm(t) {
  const { js, waitFor, shoot, problems, notes, gotoTab, confirmQuit } = t;
  const MEDIUM = 'Demo-VL Medium (Q8)';
  const row = `[...document.querySelectorAll('#panel-models tr')].find((r) => (r.querySelector('.build-name strong') || {}).textContent === ${JSON.stringify(MEDIUM)})`;
  const rowButton = (text) => `[...[${row}, ${row}.nextElementSibling].flatMap((r) => [...r.querySelectorAll('button')])].find((b) => b.textContent.trim().startsWith(${JSON.stringify(text)}))`;

  // Nothing running: quitting needs no question.
  const real = dialog.showMessageBox;
  const asked = [];
  let answer = 0;
  dialog.showMessageBox = async (...args) => {
    asked.push(args[args.length - 1]);
    return { response: answer };
  };
  try {
    if (!(await confirmQuit()) || asked.length) problems.push('quitting with nothing running should not ask');

    await gotoTab('models');
    await js(`(${rowButton('Download')} || ${rowButton('Resume')}).click(); true`);
    await waitFor(`Boolean(document.querySelector('.modal'))`, 'the download dialog');
    await js(`document.querySelector('.modal .modal-actions .button').click(); true`);
    await waitFor(`${row}.nextElementSibling.textContent.includes('left')`, 'download progress', 60000);
    await shoot('quit-01-download-running');

    answer = 0; // "Keep Downloading"
    if (await confirmQuit()) problems.push('"Keep Downloading" should cancel the quit');
    answer = 1; // "Quit"
    if (!(await confirmQuit())) problems.push('"Quit" should let the app quit');
    if (asked.length !== 2) problems.push(`expected 2 questions, saw ${asked.length}`);
    notes.push(`quit question: ${JSON.stringify(asked[0])}`);

    if (process.env.VVL_SCENARIO_NATIVE_DIALOG) {
      dialog.showMessageBox = real;
      notes.push('native dialog opened; waiting to be stopped from outside');
      await confirmQuit();
    }
  } finally {
    dialog.showMessageBox = real;
    await js(`(() => { const b = ${rowButton('Cancel')}; if (b) b.click(); return true; })()`).catch(() => {});
    await waitFor(`${row}.textContent.includes('Paused')`, 'the download to be canceled', 20000);
  }
};
