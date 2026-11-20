// Starts Electron for `npm start` and `npm run screenshots` on every platform.
// VS Code terminals export ELECTRON_RUN_AS_NODE, which makes Electron behave
// like plain Node, so it is removed from the environment first.
const { spawn } = require('child_process');
const electronBinary = require('electron');

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
delete env.ELECTRON_NO_ATTACH_CONSOLE;

const child = spawn(electronBinary, [__dirname, ...process.argv.slice(2)], { stdio: 'inherit', env });
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    if (!child.killed) child.kill(sig);
  });
}
child.on('error', (err) => {
  console.error(`Could not start Electron: ${err.message}`);
  process.exit(1);
});
child.on('exit', (code, signal) => {
  process.exit(code === null ? (signal ? 1 : 0) : code);
});
