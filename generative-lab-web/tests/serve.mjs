// Static production-artifact server for browser tests; no inference or Python.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
const root = resolve('dist/lab/browser');
const types = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png'};
createServer(async (req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = resolve(root, '.' + (name === '/' ? '/index.html' : name));
  if (!file.startsWith(root + '/')) {res.writeHead(403); res.end(); return;}
  try { const bytes = await readFile(file); res.writeHead(200, {'Content-Type':types[extname(file)] ?? 'application/octet-stream'}); res.end(bytes); }
  catch {res.writeHead(404); res.end('Not found');}
}).listen(4200, '127.0.0.1');
