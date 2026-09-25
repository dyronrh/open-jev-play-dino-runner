import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DinoSim, DINO, PHYSICS } from '../sim.js';
import { Controller } from '../agent-client.js';

const run = (seed, ticks, policy = () => ({})) => {
  const sim = new DinoSim(seed);
  while (!sim.crashed && sim.tick < ticks) sim.step(policy(sim));
  return sim;
};

test('same seed, same game', () => {
  const a = run(7, 3000), b = run(7, 3000);
  assert.equal(a.tick, b.tick);
  assert.deepEqual(a.crashed, b.crashed);
  assert.deepEqual(a.obstacles, b.obstacles);
});

test('different seeds, different obstacles', () => {
  const kinds = s => { const sim = new DinoSim(s); const k = []; while (k.length < 8 && sim.tick < 20000) { sim.step(); if (sim.crashed) { k.push(sim.crashed.kind); sim.reset(s * 1000 + k.length); } } return k.join(); };
  assert.notEqual(kinds(1), kinds(2));
});

test('doing nothing crashes into the first ground obstacle', () => {
  const sim = run(1, 5000);
  assert.ok(sim.crashed);
  assert.ok(sim.tick > PHYSICS.clearTicks);
});

// A zero-latency bot with perfect knowledge. Proves every generated gap is survivable.
const ideal = sim => {
  const o = sim.obstacles.find(o => o.x + o.width > DINO.x);
  if (!o) return {};
  const ttc = (o.x - (DINO.x + DINO.width)) / sim.speed;
  if (o.kind === 'bird_mid') return { duck: ttc < 10 };
  if (o.kind === 'bird_high') return {};
  return { jump: ttc < 10 && ttc >= 2 && !sim.dino.jumping };
};

test('the game is solvable up to top speed', () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const sim = run(seed, 130 * 60, ideal);
    assert.equal(sim.crashed, null, `seed ${seed} crashed on ${sim.crashed && sim.crashed.kind}`);
    assert.equal(sim.speed, PHYSICS.maxSpeed);
  }
});

test('jump physics: leaves the ground and lands again', () => {
  const sim = new DinoSim(1);
  sim.step({ jump: true });
  assert.ok(sim.dino.jumping && sim.dino.elev > 0);
  let peak = 0;
  while (sim.dino.jumping) { sim.step(); peak = Math.max(peak, sim.dino.elev); }
  assert.ok(peak > 60, `peak ${peak}`);
  assert.equal(sim.dino.elev, 0);
});

test('controller: jump pulses once, duck expires', () => {
  const c = new Controller();
  c.apply('jump', 0);
  assert.deepEqual(c.input(0), { jump: true, duck: false });
  assert.deepEqual(c.input(1), { jump: false, duck: false });
  c.apply('duck', 10, 5);
  assert.equal(c.input(14).duck, true);
  assert.equal(c.input(15).duck, false);
  c.apply('duck', 20, 5);
  c.apply('none', 21);
  assert.equal(c.input(22).duck, false);
});

test('observe lists only obstacles not yet passed, nearest first', () => {
  const sim = run(3, 1200, ideal);
  const obs = sim.observe();
  for (const o of obs.obstacles) assert.ok(o.x + o.width > DINO.x);
  const xs = obs.obstacles.map(o => o.x);
  assert.deepEqual(xs, [...xs].sort((a, b) => a - b));
});
