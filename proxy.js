#!/usr/bin/env node
/**
 * Combined HTTP + WebSocket proxy for Zork-Opus viewer.
 *
 * Security model (hardened 2026-07-10):
 *   - Serves ONLY an explicit allowlist of files from disk. It no longer
 *     forwards arbitrary paths to a whole-repo static server, so secrets
 *     (endpoints.json, .git, prompt logs, game_files/*) are NOT reachable.
 *   - Binds to 127.0.0.1 only. Public access must go through the basic-auth
 *     proxy (:8767) in front of the Cloudflare tunnel; nothing on the LAN can
 *     reach this port directly.
 *   - Validates the WebSocket Origin to blunt cross-site WebSocket hijacking.
 *
 * Listens on 127.0.0.1:8764. Bridges /ws to the game stream on :8765.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer, WebSocket } = require('ws');
const { URL } = require('url');

const PORT = 8764;
const BIND = '127.0.0.1';
const WS_BACKEND_PORT = 8765;
const VIEWER_DIR = path.join(__dirname);
const ROOM_IMAGES_DIR = path.join(VIEWER_DIR, 'room_images');

// Only these origins may open the game WebSocket. A browser on any other site
// cannot forge Origin, so this stops cross-site WebSocket hijacking even if the
// victim's browser has cached the tunnel's basic-auth credentials.
const ALLOWED_WS_ORIGINS = new Set([
  'https://zork.schuyler.ai',
  'http://localhost:8764',
  'http://127.0.0.1:8764',
]);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.json': 'application/json',
};

// Exact-path allowlist: request path -> file on disk (relative to VIEWER_DIR).
const EXACT_FILES = {
  '/': 'viewer.html',
  '/viewer.html': 'viewer.html',
  '/favicon.png': 'favicon.png',
  '/current_state.json': 'current_state.json',
};

function sendFile(res, absPath) {
  fs.readFile(absPath, (err, content) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    const ext = path.extname(absPath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    res.end(content);
  });
}

const server = http.createServer((req, res) => {
  let url;
  try {
    url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  } catch (e) {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end('Bad request');
    return;
  }
  const pathname = url.pathname;

  if (pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  // Exact-file allowlist (viewer.html, favicon, current_state.json).
  if (Object.prototype.hasOwnProperty.call(EXACT_FILES, pathname)) {
    sendFile(res, path.join(VIEWER_DIR, EXACT_FILES[pathname]));
    return;
  }

  // room_images/<file> — the only prefix we serve. Resolve and confirm the
  // result stays inside room_images/ so "../" traversal cannot escape it.
  if (pathname.startsWith('/room_images/')) {
    const rel = decodeURIComponent(pathname.slice('/room_images/'.length));
    const abs = path.normalize(path.join(ROOM_IMAGES_DIR, rel));
    if (abs === ROOM_IMAGES_DIR || !abs.startsWith(ROOM_IMAGES_DIR + path.sep)) {
      res.writeHead(403, { 'Content-Type': 'text/plain' });
      res.end('Forbidden');
      return;
    }
    sendFile(res, abs);
    return;
  }

  // Everything else — endpoints.json, .git, game_files/, prompt logs — 404.
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found');
});

// --- WebSocket bridge to the game stream, with Origin validation ---
const wss = new WebSocketServer({
  server,
  path: '/ws',
  verifyClient: ({ origin }) => {
    // Browsers always send Origin; a mismatched one is a cross-site attempt.
    // A missing Origin means a non-browser client (local tooling), which can't
    // mount the cross-site attack this guard is for, so we allow it.
    if (!origin) return true;
    return ALLOWED_WS_ORIGINS.has(origin);
  },
});

wss.on('connection', (clientWs) => {
  console.log('[ws] Client connected, bridging to localhost:' + WS_BACKEND_PORT);
  const backendWs = new WebSocket('ws://localhost:' + WS_BACKEND_PORT + '/ws');

  backendWs.on('open', () => {
    console.log('[ws] Backend connected');
    clientWs.on('message', (data, isBinary) => backendWs.send(data, { binary: isBinary }));
    backendWs.on('message', (data, isBinary) => clientWs.send(data, { binary: isBinary }));
    clientWs.on('close', () => { console.log('[ws] Client disconnected'); backendWs.close(); });
    backendWs.on('close', () => { console.log('[ws] Backend disconnected'); clientWs.close(); });
    backendWs.on('error', (err) => { console.error('[ws] Backend error:', err.message); clientWs.close(); });
  });

  backendWs.on('error', (err) => {
    console.error('[ws] Failed to connect backend:', err.message);
    clientWs.close(1011, 'Backend unavailable');
  });
});

server.listen(PORT, BIND, () => {
  console.log(`Zork-Opus proxy listening on ${BIND}:${PORT}`);
  console.log('  Serving allowlist: /, /viewer.html, /favicon.png, /current_state.json, /room_images/*');
  console.log(`  Bridging /ws to localhost:${WS_BACKEND_PORT} (Origin-validated)`);
});

process.on('SIGTERM', () => { console.log('Shutting down...'); wss.close(); server.close(); });
process.on('SIGINT', () => { console.log('Shutting down...'); wss.close(); server.close(); });
