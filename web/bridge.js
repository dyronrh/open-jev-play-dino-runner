// Connects the original Chrome Dino game to the decision server.
//
// The game (web/original/index.js) is loaded unmodified. This module only:
//   - patches two cosmetic prototype methods from outside (arcade-mode full-window scaling, sound
//     when muted) and hooks init/restart to seed the obstacle sequence;
//   - reads the game state every animation frame and sends it over the WebSocket;
//   - presses keys through the game's own handlers when an answer comes back.
//
// URL parameters:
//   ?autoplay=1&seed=7&max_seconds=120   headless episode mode (used by the Python Arena): starts
//                                        immediately, plays one episode, publishes window.__episode
//   ?seed=N  first seed      ?mute=1  no sound
import { AgentClient, percentile } from './agent-client.js';
import { Driver, obstacleKind, observe, score, seedObstacles } from './dino-env.js';

const params = new URLSearchParams(location.search);
const AUTOPLAY = params.get('autoplay') === '1';
const MAX_SECONDS = Number(params.get('max_seconds') || 0);
const MUTE = AUTOPLAY || params.get('mute') === '1';
const $ = id => document.getElementById(id);
const seedInput = $('seed');
if (params.get('seed')) seedInput.value = params.get('seed');
const currentSeed = () => Number(seedInput.value) || 1;
const mode = () => (AUTOPLAY ? 'agent' : $('mode').value);

// ---- Patches, applied before index.js creates the Runner on DOMContentLoaded ----
const Runner = window.Runner;
Runner.prototype.setArcadeMode = function () {}; // keep the game inside this page instead of full-window
if (MUTE) Runner.prototype.playSound = function () {};
const originalInit = Runner.prototype.init;
Runner.prototype.init = function () {
  originalInit.call(this);
  seedObstacles(this, currentSeed());
  onReady(this);
};
const originalRestart = Runner.prototype.restart;
Runner.prototype.restart = function () {
  seedObstacles(this, currentSeed()); // every run of a seed meets the same obstacles
  return originalRestart.call(this);
};

// ---- Episode state ----
let runner, driver, client;
let tick = 0, startedAt = 0, finished = false, restartTimer = null;
let recent = [];

function onReady(r) {
  runner = r;
  driver = new Driver(runner);
  client = new AgentClient(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`, {
    reconnect: !AUTOPLAY,
    onStatus: s => {
      const el = $('status');
      el.textContent = s === 'ready' ? `ready · brain: ${client.hello.brain.name}` : s;
      el.className = s === 'ready' ? 'ready' : /error|mismatch|disconnected/.test(s) ? 'bad' : '';
      if (s === 'ready' && mode() === 'agent') start();
    },
    onAct: msg => {
      if (mode() !== 'agent' || !runner.playing || runner.crashed || !client.fresh(msg, tick)) return;
      driver.apply(msg.action, tick, msg.hold_ticks);
      recent.push({ tick: msg.tick, applied_at: tick, action: msg.action, text: msg.text, answers: msg.answers, latency_ms: msg.latency_ms });
      if (recent.length > 5) recent.shift();
      showDecision(msg);
    },
  });
  client.connect().then(ok => { if (!ok && AUTOPLAY) publish({ seed: currentSeed(), error: 'could not connect to the decision server' }); });
  requestAnimationFrame(frame);
}

// Start (or restart) an episode with the current seed.
function start() {
  clearTimeout(restartTimer);
  tick = 0; finished = false; recent = []; startedAt = performance.now();
  if (runner.crashed) runner.restart();
  else if (!runner.playing) driver.key('keydown', 32); // the first jump starts the game, as with the space bar
}

function frame() {
  if (runner.playing && !runner.crashed) {
    tick++;
    driver.tick(tick);
    if (mode() === 'agent') client.maybeSend(observe(runner, tick));
    if (MAX_SECONDS && performance.now() - startedAt >= MAX_SECONDS * 1000) end('timeout');
  }
  if (runner.crashed && !finished) end('crash');
  hud();
  requestAnimationFrame(frame);
}

function end(outcome) {
  finished = true;
  if (outcome === 'timeout') runner.stop();
  if (mode() !== 'agent') return;
  const s = client.stats, rtts = s.rtts;
  const summary = {
    seed: currentSeed(),
    score: score(runner),
    ticks: tick,
    seconds: +((performance.now() - startedAt) / 1000).toFixed(2),
    speed: +runner.currentSpeed.toFixed(2),
    outcome,
    crashed_on: outcome === 'crash' && runner.horizon.obstacles[0] ? obstacleKind(runner.horizon.obstacles[0]) : null,
    decisions: s.applied, stale: s.stale, timeouts: s.timeouts,
    rtt_p50_ms: percentile(rtts, 0.5), rtt_p95_ms: percentile(rtts, 0.95), rtt_p99_ms: percentile(rtts, 0.99),
    last_decisions: outcome === 'crash' ? recent : [],
  };
  client.sendEpisodeEnd({ ...summary, source: AUTOPLAY ? 'headless' : 'browser' });
  if (AUTOPLAY) return publish(summary);
  restartTimer = setTimeout(() => { seedInput.value = currentSeed() + 1; start(); }, 1500); // keep playing new seeds
}

function publish(summary) {
  window.__episode = summary;
  setTimeout(() => client && client.close(), 50);
}

// ---- UI ----
$('restart').onclick = () => { if (mode() === 'agent') start(); else { if (runner.crashed) runner.restart(); } };
$('mode').onchange = () => { if (mode() === 'agent' && client.ready) start(); };

function showDecision(msg) {
  $('text').textContent = msg.text;
  $('action').textContent = msg.action.toUpperCase();
  $('reason').textContent = msg.reason;
  $('answers').innerHTML = Object.entries(msg.answers).map(([q, a]) =>
    `<div class="q"><span>${q}: ${a.choice}</span><div class="meter"><i style="width:${Math.round(a.p * 100)}%"></i></div></div>`).join('');
}

function hud() {
  if (AUTOPLAY) return;
  $('score').textContent = score(runner);
  $('speed').textContent = runner.currentSpeed.toFixed(2);
  const r = client.stats.rtts.slice(-300);
  $('rtt').textContent = r.length ? `${percentile(r, 0.5).toFixed(1)} / ${percentile(r, 0.95).toFixed(1)} ms` : '–';
  $('decisions').textContent = `${client.stats.applied} / ${client.stats.stale}`;
}
