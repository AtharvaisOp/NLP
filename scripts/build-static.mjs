/** Publish only website assets: Python, datasets, weights and secrets stay out. */
import { cp, mkdir, rm, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'dist');
if (path.dirname(output) !== root || path.basename(output) !== 'dist') throw new Error('Invalid output directory');
await rm(output, { recursive: true, force: true });
await mkdir(output);
const directories = ['ai-usage', 'analyzer', 'applications', 'assets', 'category', 'compare', 'concept', 'design-system', 'favicon_io', 'references', 'reflection', 'repository', 'research', 'sustainability', 'team', 'workflows'];
for (const directory of directories) await cp(path.join(root, directory), path.join(output, directory), { recursive: true });
for (const file of await readdir(root)) {
  if (/\.(html|png|ico|webmanifest)$/.test(file)) await cp(path.join(root, file), path.join(output, file));
}
const base = process.env.MAHAPULSE_API_BASE_URL || 'http://localhost:8000';
const parsed = new URL(base);
if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error('API URL must be a public HTTP(S) base without credentials');
await writeFile(path.join(output, 'assets/js/runtime-config.js'), `window.MAHAPULSE_STATIC_CONFIG = ${JSON.stringify({ API_BASE_URL: base.replace(/\/+$/, ''), USE_MOCK_API: false, MAX_UPLOAD_BYTES: 5000000, MAX_BATCH_ROWS: 1000, DEFAULT_TEXT_COLUMN: 'text', BATCH_TIMEOUT_MS: 300000 })};\n`);
console.log(`Static website built in dist; API: ${parsed.origin}`);
