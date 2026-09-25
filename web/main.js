// Browser entry point: runs the simulation at a fixed 60 Hz, draws it, and lets either the human or
// the decision server play.
import { DinoSim, TICK_HZ, WIDTH, HEIGHT, GROUND_Y, DINO } from './sim.js';
import { AgentClient, Controller, percentile } from './agent-client.js';

const $ = id => document.getElementById(id);
const canvas = $('stage'), ctx = canvas.getContext('2d');
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let sim = new DinoSim(Number($('seed').value));
let controller = new Controller();
let best = 0;
const keys = { jump: false, duck: false };

const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const client = new AgentClient(wsUrl, {
  onStatus: s => { $('status').textContent = s === 'ready' ? `ready · brain: ${client.hello.brain.name}` : s; $('status').className = s === 'ready' ? 'ready' : /error|mismatch|disconnected/.test(s) ? 'bad' : ''; },
  onAct: msg => {
    if (mode() !== 'agent' || !client.fresh(msg, sim.tick)) return;
    controller.apply(msg.action, sim.tick, msg.hold_ticks);
    showDecision(msg);
  },
});
client.connect();

const mode = () => $('mode').value;

function restart() {
  if (sim.tick > 0 && mode() === 'agent') client.sendEpisodeEnd({ seed: sim.seed, score: sim.score, ticks: sim.tick, outcome: sim.crashed ? 'crash' : 'restart', crashed_on: sim.crashed && sim.crashed.kind, source: 'browser' });
  sim = new DinoSim(Number($('seed').value) || 1);
  controller = new Controller();
}
$('restart').onclick = restart;
$('mode').onchange = restart;

addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space' || e.code === 'ArrowUp') { keys.jump = true; e.preventDefault(); }
  if (e.code === 'ArrowDown') { keys.duck = true; e.preventDefault(); }
  if (e.code === 'KeyR') restart();
});
addEventListener('keyup', e => {
  if (e.code === 'ArrowDown') keys.duck = false;
});

function step() {
  if (sim.crashed) return;
  const input = mode() === 'human' ? { jump: keys.jump, duck: keys.duck } : controller.input(sim.tick);
  keys.jump = false;
  sim.step(input);
  if (mode() === 'agent') client.maybeSend(sim.observe());
  if (sim.crashed) {
    best = Math.max(best, sim.score);
    if (mode() === 'agent') {
      client.sendEpisodeEnd({ seed: sim.seed, score: sim.score, ticks: sim.tick, outcome: 'crash', crashed_on: sim.crashed.kind, source: 'browser' });
      setTimeout(() => { $('seed').value = sim.seed + 1; restart(); }, 1500); // keep playing new seeds
    }
  }
}

function showDecision(msg) {
  $('text').textContent = msg.text;
  $('action').textContent = msg.action.toUpperCase();
  $('reason').textContent = msg.reason;
  $('answers').innerHTML = Object.entries(msg.answers).map(([q, a]) =>
    `<div class="q"><span>${q}: ${a.choice}</span><div class="meter"><i style="width:${Math.round(a.p * 100)}%"></i></div></div>`).join('');
}

// Drawing: flat shapes in the style of the original.
function draw() {
  const fg = css('--fg'), bg = css('--bg');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = fg;
  ctx.fillRect(0, GROUND_Y, WIDTH, 1);
  for (let x = -(sim.distance % 37); x < WIDTH; x += 37) ctx.fillRect(x, GROUND_Y + 4 + (Math.abs(x | 0) % 3) * 3, 3, 1);

  for (const o of sim.obstacles) {
    const top = GROUND_Y - o.elev - o.height;
    if (o.kind.startsWith('cactus')) {
      const n = Number(o.kind.slice(-1)), w = o.width / n;
      for (let i = 0; i < n; i++) {
        const x = o.x + i * w;
        ctx.fillRect(x + w * 0.35, top, w * 0.3, o.height);
        ctx.fillRect(x + w * 0.05, top + o.height * 0.3, w * 0.2, o.height * 0.3);
        ctx.fillRect(x + w * 0.75, top + o.height * 0.2, w * 0.2, o.height * 0.35);
      }
    } else {
      const flap = Math.floor(sim.tick / 10) % 2;
      ctx.fillRect(o.x, top + 16, o.width, 8);
      ctx.fillRect(o.x + 10, flap ? top : top + 22, 18, flap ? 18 : 16);
      ctx.fillRect(o.x - 4, top + 14, 10, 5);
    }
  }

  const d = sim.dino, w = d.ducking ? DINO.duckWidth : DINO.width, h = d.ducking ? DINO.duckHeight : DINO.height;
  const x = DINO.x, y = GROUND_Y - d.elev - h;
  ctx.fillStyle = fg;
  ctx.fillRect(x, y + h * 0.25, w * 0.62, h * 0.5);          // body
  ctx.fillRect(x + w * 0.45, y, w * 0.55, h * 0.36);           // head
  const leg = Math.floor(sim.tick / 6) % 2 && !d.jumping;
  ctx.fillRect(x + w * 0.12, y + h * 0.75, 5, h * (leg ? 0.25 : 0.18));
  ctx.fillRect(x + w * 0.38, y + h * 0.75, 5, h * (leg ? 0.18 : 0.25));
  ctx.fillStyle = bg;
  ctx.fillRect(x + w * 0.62, y + 4, 4, 4);                     // eye

  if (sim.crashed) {
    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.font = '16px ui-monospace, monospace';
    ctx.fillText('G A M E   O V E R', WIDTH / 2, 55);
  }
}

function hud() {
  $('score').textContent = sim.score;
  $('best').textContent = Math.max(best, sim.score);
  $('speed').textContent = sim.speed.toFixed(2);
  const r = client.stats.rtts.slice(-300);
  $('rtt').textContent = r.length ? `${percentile(r, 0.5).toFixed(1)} / ${percentile(r, 0.95).toFixed(1)} ms` : '–';
  $('decisions').textContent = `${client.stats.applied} / ${client.stats.stale}`;
}

// Fixed timestep: the simulation always advances at 60 Hz whatever the display refresh rate.
const dt = 1000 / TICK_HZ;
let last = performance.now(), acc = 0;
function frame(now) {
  acc += Math.min(now - last, 250);
  last = now;
  while (acc >= dt) { step(); acc -= dt; }
  draw();
  hud();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
