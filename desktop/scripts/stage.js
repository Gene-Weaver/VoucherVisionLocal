// Puts everything the installer carries besides Electron into desktop/.stage/:
//
//   .stage/backend/           the Python app: app.py, vvl/, textcollage/, ui/, built-in prompts
//   .stage/python/            what uv needs to rebuild the environment: pyproject.toml, uv.lock,
//                             and python-version (the pinned Python, from .python-version)
//   .stage/uv/<os>-<arch>/    the pinned uv binary for each build target (checked by sha256)
//
// The backend folder gets no pyproject.toml on purpose: the backend treats a
// pyproject.toml beside app.py as a source checkout and would then keep the
// user's keys and settings inside the (read-only, replaced-on-update) app.
//
// Usage: node scripts/stage.js [--targets linux-x64,mac-arm64,mac-x64]
// (default: this computer's platform and CPU).
const { execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');
const path = require('path');

const desktop = path.resolve(__dirname, '..');
const repo = path.resolve(desktop, '..');
const stage = path.join(desktop, '.stage');
const downloads = path.join(stage, 'downloads');
const uvPin = JSON.parse(fs.readFileSync(path.join(desktop, 'bundled-uv.json'), 'utf8'));

// Folders and files of the repo that the installed app needs.
const BACKEND = ['app.py', 'VERSION', 'LICENSE', 'vvl/', 'textcollage/', 'ui/', 'prompts/builtin/'];

function thisTarget() {
  const os = { linux: 'linux', darwin: 'mac', win32: 'win' }[process.platform];
  return `${os}-${process.arch}`;
}

function parseTargets() {
  const i = process.argv.indexOf('--targets');
  const list = i >= 0 && process.argv[i + 1] ? process.argv[i + 1].split(',') : [thisTarget()];
  for (const t of list) {
    if (!uvPin.targets[t]) {
      console.error(`Unknown target "${t}". Known: ${Object.keys(uvPin.targets).join(', ')}`);
      process.exit(1);
    }
  }
  return list;
}

/** Files git would commit (tracked, plus new files that are not ignored) under the given paths. */
function repoFiles(paths) {
  const out = execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', ...paths], { cwd: repo });
  return out.toString('utf8').split('\0').filter((f) => f && fs.existsSync(path.join(repo, f)))
    .filter((f) => !/(^|\/)__pycache__\//.test(f) && !f.endsWith('.pyc'));
}

function copyInto(destRoot, files) {
  for (const f of files) {
    const dest = path.join(destRoot, f);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(path.join(repo, f), dest);
  }
}

function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function download(url, dest, redirects = 5) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'vouchervisionlocal-build' } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects > 0) {
        res.resume();
        resolve(download(new URL(res.headers.location, url).href, dest, redirects - 1));
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      const tmp = `${dest}.partial`;
      const out = fs.createWriteStream(tmp);
      res.pipe(out);
      out.on('finish', () => out.close(() => {
        fs.renameSync(tmp, dest);
        resolve();
      }));
      out.on('error', reject);
    }).on('error', reject);
  });
}

async function stageUv(target) {
  const pin = uvPin.targets[target];
  const archive = path.join(downloads, `uv-${uvPin.version}-${pin.file}`);
  fs.mkdirSync(downloads, { recursive: true });
  if (!fs.existsSync(archive) || sha256(archive) !== pin.sha256) {
    const url = uvPin.url.replace('{version}', uvPin.version).replace('{file}', pin.file);
    console.log(`[stage] downloading ${url}`);
    await download(url, archive);
  }
  const got = sha256(archive);
  if (got !== pin.sha256) {
    fs.rmSync(archive, { force: true });
    throw new Error(`sha256 mismatch for ${pin.file}: expected ${pin.sha256}, got ${got}`);
  }
  const outDir = path.join(stage, 'uv', target);
  const work = path.join(stage, 'uv', `.${target}-extract`);
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.rmSync(work, { recursive: true, force: true });
  fs.mkdirSync(work, { recursive: true });
  const exe = target.startsWith('win-') ? 'uv.exe' : 'uv';
  if (pin.file.endsWith('.zip')) execFileSync('unzip', ['-q', archive, '-d', work]);
  else execFileSync('tar', ['-xzf', archive, '-C', work]);
  const found = execFileSync('find', [work, '-type', 'f', '-name', exe]).toString().trim().split('\n').filter(Boolean);
  if (found.length !== 1) throw new Error(`expected one ${exe} in ${pin.file}, found ${found.length}`);
  fs.mkdirSync(outDir, { recursive: true });
  fs.copyFileSync(found[0], path.join(outDir, exe));
  fs.chmodSync(path.join(outDir, exe), 0o755);
  fs.writeFileSync(path.join(outDir, 'VERSION'), `${uvPin.version}\n`);
  fs.rmSync(work, { recursive: true, force: true });
  console.log(`[stage] uv ${uvPin.version} for ${target}`);
}

async function main() {
  const targets = parseTargets();

  // Backend: a clean copy every time, so a deleted source file never lingers in a build.
  const backend = path.join(stage, 'backend');
  fs.rmSync(backend, { recursive: true, force: true });
  const files = repoFiles(BACKEND);
  for (const must of ['app.py', 'VERSION', 'vvl/cli.py', 'ui/index.html', 'textcollage/models/openvino/best.bin']) {
    if (!files.includes(must)) throw new Error(`${must} is missing from the backend file list`);
  }
  if (files.some((f) => /(^|\/)PRIVATE[^/]*\.yaml$/.test(f) && !f.endsWith('PRIVATE.example.yaml'))) {
    throw new Error('a PRIVATE*.yaml file would be packaged');
  }
  copyInto(backend, files);
  console.log(`[stage] backend: ${files.length} files`);

  // What uv needs to build the environment. No dot-files (some packagers skip them).
  const python = path.join(stage, 'python');
  fs.rmSync(python, { recursive: true, force: true });
  copyInto(python, ['pyproject.toml', 'uv.lock', 'README.md', 'LICENSE', 'VERSION']);
  const pyVersion = fs.readFileSync(path.join(repo, '.python-version'), 'utf8').trim();
  if (!/^3\.\d+(\.\d+)?$/.test(pyVersion)) throw new Error(`.python-version looks wrong: "${pyVersion}"`);
  fs.writeFileSync(path.join(python, 'python-version'), `${pyVersion}\n`);
  console.log(`[stage] python project (Python ${pyVersion})`);

  for (const t of targets) await stageUv(t);
}

main().catch((err) => {
  console.error(`[stage] ${err.message}`);
  process.exit(1);
});
