import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const testDir = path.join(root, 'frontend-dev', 'tests');
const tests = fs.readdirSync(testDir).filter(name => /\.js$/.test(name)).sort();
let failed = 0;
for (const test of tests) {
  const result = spawnSync(process.execPath, [path.join(testDir, test)], {
    cwd: root, encoding: 'utf8', timeout: 120000,
  });
  process.stdout.write(result.stdout || '');
  process.stderr.write(result.stderr || '');
  if (result.error || result.status !== 0) {
    failed += 1;
    console.error(`${test} failed: ${result.error?.message || `exit ${result.status}`}`);
  }
}
console.log(`Frontend Node tests: ${tests.length - failed}/${tests.length} files passed.`);
if (failed || !tests.length) process.exitCode = 1;
