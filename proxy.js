#!/usr/bin/env node
/**
 * Combined HTTP + WebSocket proxy for Zork-Opus viewer.
 * Serves viewer files from http://localhost:8766
 * Bridges WebSocket connections at /ws to localhost:8765
 * Listens on port 8764 (single port for Cloudflare tunnel)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const net = require('net');
const { WebSocketServer, WebSocket } = require('ws');
const { URL } = require('url');

const PORT = 8764;
const VIEWER_PORT = 8766;
const WS_BACKEND_PORT = 8765;
const VIEWER_DIR = path.join(__dirname);

// MIME types for static files
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

// --- HTTP server (static file serving + proxy) ---
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  // Proxy health check
  if (url.pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  // Serve static viewer.html
  if (url.pathname === '/' || url.pathname === '/viewer.html') {
    const filePath = path.join(VIEWER_DIR, 'viewer.html');
    try {
      const content = fs.readFileSync(filePath);
      res.writeHead(200, { 'Content-Type': MIME['.html'] });
      res.end(content);
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('viewer.html not found');
    }
    return;
  }

  // Proxy everything else to the http.server on VIEWER_PORT
  try {
    const proxyReq = http.request({
      hostname: 'localhost',
      port: VIEWER_PORT,
      path: req.url,
      method: req.method,
      headers: req.headers,
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    });
    proxyReq.on('error', () => {
      res.writeHead(502, { 'Content-Type': 'text/plain' });
      res.end('Backend unavailable on port ' + VIEWER_PORT);
    });
    req.pipe(proxyReq);
  } catch (e) {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('Proxy error: ' + e.message);
  }
});

// --- WebSocket server (bridge to backend) ---
const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (clientWs, req) => {
  console.log('[ws] Client connected, bridging to localhost:' + WS_BACKEND_PORT);

  const backendWs = new WebSocket('ws://localhost:' + WS_BACKEND_PORT + '/ws');

  backendWs.on('open', () => {
    console.log('[ws] Backend connected');

    clientWs.on('message', (data, isBinary) => {
      backendWs.send(data, { binary: isBinary });
    });

    backendWs.on('message', (data, isBinary) => {
      clientWs.send(data, { binary: isBinary });
    });

    clientWs.on('close', () => {
      console.log('[ws] Client disconnected');
      backendWs.close();
    });

    backendWs.on('close', () => {
      console.log('[ws] Backend disconnected');
      clientWs.close();
    });

    backendWs.on('error', (err) => {
      console.error('[ws] Backend error:', err.message);
      clientWs.close();
    });
  });

  backendWs.on('error', (err) => {
    console.error('[ws] Failed to connect backend:', err.message);
    clientWs.close(1011, 'Backend unavailable');
  });
});

// --- Start ---
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Zork-Opus proxy listening on port ${PORT}`);
  console.log(`  Serving viewer from localhost:${VIEWER_PORT}`);
  console.log(`  Bridging /ws to localhost:${WS_BACKEND_PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down...');
  wss.close();
  server.close();
});

process.on('SIGINT', () => {
  console.log('Shutting down...');
  wss.close();
  server.close();
});
