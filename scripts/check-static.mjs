import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const routeDirs = [
  'ai-usage', 'analyzer', 'applications', 'category', 'compare', 'concept',
  'design-system', 'references', 'reflection', 'repository', 'research',
  'sustainability', 'team', 'workflows',
];
const htmlFiles = ['index.html', '404.html', ...routeDirs.map(dir => `${dir}/index.html`)];
const errors = [];
let links = 0;
let inlineScripts = 0;
let jsFiles = 0;

function checkLink(filename, value) {
  const url = value.trim();
  if (!url || url.startsWith('#') || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(url)) return;
  let localPath;
  try { localPath = decodeURIComponent(url.split(/[?#]/, 1)[0]); }
  catch { errors.push(`${filename}: invalid encoded local URL ${url}`); return; }
  if (!localPath) return;
  const target = path.resolve(localPath.startsWith('/') ? root : path.dirname(path.join(root, filename)), `.${localPath.startsWith('/') ? localPath : path.sep + localPath}`);
  if (target !== root && !target.startsWith(root + path.sep)) {
    errors.push(`${filename}: local URL escapes published site: ${url}`);
    return;
  }
  links += 1;
  if (!fs.existsSync(target)) errors.push(`${filename}: missing local target ${url}`);
  else if (fs.statSync(target).isDirectory() && !fs.existsSync(path.join(target, 'index.html'))) {
    errors.push(`${filename}: directory URL has no index.html: ${url}`);
  }
}

for (const filename of htmlFiles) {
  assert.ok(fs.existsSync(path.join(root, filename)), `Required static route missing: ${filename}`);
  const html = fs.readFileSync(path.join(root, filename), 'utf8');
  for (const tag of html.matchAll(/<(?:a|link|script|img|source|iframe|form)\b[^>]*>/gi)) {
    for (const attribute of tag[0].matchAll(/\b(?:href|src|action)\s*=\s*(["'])(.*?)\1/gi)) {
      checkLink(filename, attribute[2]);
    }
  }
  for (const script of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)) {
    if (/\bsrc\s*=/i.test(script[1]) || !script[2].trim()) continue;
    const type = script[1].match(/\btype\s*=\s*(["'])(.*?)\1/i)?.[2];
    if (type && !/^(?:text|application)\/(?:java|ecma)script$/i.test(type)) continue;
    inlineScripts += 1;
    try { new vm.Script(script[2], { filename: `${filename}:inline-${inlineScripts}` }); }
    catch (error) { errors.push(error.message); }
  }
}

function syntaxTree(dirname) {
  for (const entry of fs.readdirSync(dirname, { withFileTypes: true })) {
    const filename = path.join(dirname, entry.name);
    if (entry.isDirectory()) syntaxTree(filename);
    else if (/\.(?:js|mjs)$/.test(entry.name)) {
      jsFiles += 1;
      const result = spawnSync(process.execPath, ['--check', filename], { encoding: 'utf8' });
      if (result.status !== 0) errors.push(result.stderr || `${filename}: syntax check failed`);
    }
  }
}
for (const dirname of ['assets/js', 'frontend-dev', 'scripts']) syntaxTree(path.join(root, dirname));

if (errors.length) {
  console.error(errors.join('\n'));
  process.exitCode = 1;
} else {
  console.log(`Static checks passed: ${htmlFiles.length} HTML routes, ${links} local links/assets, ${inlineScripts} inline scripts, ${jsFiles} JS files.`);
}
