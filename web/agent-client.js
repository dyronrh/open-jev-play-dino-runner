// WebSocket client for the decision server, used by web/bridge.js.
//
// Flow control: at most one observation is in flight. The next one is sent only after the answer
// arrives (or times out), so the server never builds a queue and every answer is about a fresh
// state. Answers older than `maxAgeTicks` are dropped instead of applied.

export const PROTOCOL_VERSION = 1;
const now = () => (globalThis.performance ? performance.now() : Date.now());

export class AgentClient {
  constructor(url, { WebSocketImpl = globalThis.WebSocket, reconnect = true, timeoutMs = 1000, onAct, onStatus } = {}) {
    this.url = url;
    this.WebSocketImpl = WebSocketImpl;
    this.reconnect = reconnect;
    this.timeoutMs = timeoutMs;
    this.onAct = onAct || (() => {});
    this.onStatus = onStatus || (() => {});
    this.ws = null;
    this.hello = null;
    this.inflight = null;
    this.lastRtt = null;
    this.backoff = 250;
    this.closed = false;
    this.stats = { sent: 0, applied: 0, stale: 0, timeouts: 0, errors: 0, rtts: [] };
  }

  get ready() {
    return !!(this.ws && this.ws.readyState === 1 && this.hello);
  }

  connect() {
    this.closed = false;
    return new Promise(resolve => {
      const ws = new this.WebSocketImpl(this.url);
      this.ws = ws;
      this.onStatus('connecting');
      ws.onmessage = ev => this.handle(JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString()), resolve);
      ws.onclose = () => {
        this.hello = null;
        this.inflight = null;
        this.onStatus('disconnected');
        if (this.reconnect && !this.closed) {
          setTimeout(() => this.connect(), this.backoff);
          this.backoff = Math.min(this.backoff * 2, 5000);
        }
        resolve(false);
      };
      ws.onerror = () => {}; // onclose follows and handles it
    });
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }

  handle(msg, resolveConnect) {
    if (msg.type === 'hello') {
      if (msg.v !== PROTOCOL_VERSION) { this.onStatus(`protocol mismatch: server v${msg.v}`); this.close(); return; }
      this.hello = msg;
      this.backoff = 250;
      this.onStatus('ready');
      resolveConnect(true);
    } else if (msg.type === 'act') {
      if (!this.inflight || msg.tick !== this.inflight.tick) return; // an answer we already gave up on
      const rtt = now() - this.inflight.t0;
      this.inflight = null;
      this.lastRtt = rtt;
      this.stats.rtts.push(rtt);
      if (this.stats.rtts.length > 5000) this.stats.rtts.shift();
      this.onAct(msg, rtt);
    } else if (msg.type === 'error') {
      this.inflight = null;
      this.stats.errors++;
      this.onStatus(`server error: ${msg.error}`);
    }
  }

  // Call once per tick. Sends the observation if nothing is in flight; returns true when sent.
  maybeSend(obs) {
    if (!this.ready) return false;
    if (this.inflight) {
      if (now() - this.inflight.t0 < this.timeoutMs) return false;
      this.stats.timeouts++;
      this.inflight = null;
    }
    this.inflight = { tick: obs.tick, t0: now() };
    this.stats.sent++;
    this.ws.send(JSON.stringify({ v: PROTOCOL_VERSION, type: 'obs', rtt_ms: this.lastRtt, ...obs }));
    return true;
  }

  // True when an answer is fresh enough to act on.
  fresh(msg, currentTick) {
    const maxAge = (this.hello && this.hello.max_age_ticks) || 12;
    if (currentTick - msg.tick > maxAge) { this.stats.stale++; return false; }
    this.stats.applied++;
    return true;
  }

  sendEpisodeEnd(summary) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ v: PROTOCOL_VERSION, type: 'episode_end', ...summary }));
  }
}

export function percentile(values, p) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  return +s[Math.min(s.length - 1, Math.floor(p * s.length))].toFixed(2);
}
