// Copies the one version number (repo-root VERSION) into desktop/package.json
// and package-lock.json, so the installer, the updater and the Python backend
// all report the same version. Run by `npm run stage` before every build.
const fs = require('fs');
const path = require('path');

const desktop = path.resolve(__dirname, '..');
const version = fs.readFileSync(path.join(desktop, '..', 'VERSION'), 'utf8').trim();
if (!/^\d+\.\d+\.\d+$/.test(version)) {
  console.error(`VERSION must look like 1.2.3, found "${version}"`);
  process.exit(1);
}

function update(file, apply) {
  const full = path.join(desktop, file);
  const before = fs.readFileSync(full, 'utf8');
  const data = JSON.parse(before);
  apply(data);
  const after = `${JSON.stringify(data, null, 2)}\n`;
  if (after !== before) fs.writeFileSync(full, after);
  return after !== before;
}

const changed = [
  update('package.json', (p) => { p.version = version; }),
  update('package-lock.json', (l) => {
    l.version = version;
    if (l.packages && l.packages['']) l.packages[''].version = version;
  }),
].some(Boolean);
console.log(`[version] ${version}${changed ? ' (package.json updated)' : ''}`);
