// Scenario: the engine fails to start and the Engine tab shows why.
// Fake backend only, started so that its engine runs out of memory:
//
//   FAKE_LLAMA_MODE=oom_on_start uv run python tools/dev_fake_backend.py --root <scratch> \
//     --preinstall-engine --preinstall-model

module.exports = async function engineError(t) {
  const { js, waitFor, shoot, problems, gotoTab, setTheme } = t;
  for (const theme of ['light', 'dark']) {
    await setTheme(theme);
    await gotoTab('engine');
    await js(`document.getElementById('engineSection').scrollIntoView(); true`);
    await waitFor(`!document.getElementById('engineStart').disabled`, 'the Start button');
    await js(`document.getElementById('engineStart').click(); true`);
    await waitFor(`document.querySelector('.engine-status .badge').textContent !== 'Error'`, 'the engine to begin starting', 10000);
    const failed = await waitFor(`document.querySelector('.engine-status .badge').textContent === 'Error'`, 'the engine error state', 60000);
    if (failed && !(await js(`Boolean(document.querySelector('#engineSection .job-problem'))`))) problems.push('the engine error is not shown');
    await js(`(() => { const d = document.querySelector('#engineSection .job-problem details'); if (d) d.open = true; return true; })()`);
    await shoot(`engine-error-${theme}`);
  }
  await setTheme('light');
};
