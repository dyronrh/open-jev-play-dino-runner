// Unit tests for the adapter, against a fake runner shaped like the original game's objects.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Driver, obstacleKind, observe, rng, score, seedObstacles, KEY } from '../dino-env.js';

const small = size => ({ typeConfig: { type: 'CACTUS_SMALL', height: 35 }, size, xPos: 300, yPos: 105, width: 17 * size });
const large = size => ({ typeConfig: { type: 'CACTUS_LARGE', height: 50 }, size, xPos: 300, yPos: 90, width: 25 * size });
const bird = yPos => ({ typeConfig: { type: 'PTERODACTYL', height: 40 }, size: 1, xPos: 300, yPos, width: 46 });

function fakeRunner({ obstacles = [], jumping = false, ducking = false, yPos = 93 } = {}) {
  const keys = [];
  const runner = {
    currentSpeed: 8.123456,
    distanceRan: 1000,
    distanceMeter: { getActualDistance: d => Math.round(d * 0.025) },
    tRex: { xPos: 23, yPos, groundYPos: 93, jumping, ducking, config: { WIDTH: 44, WIDTH_DUCK: 59 } },
    horizon: { obstacles, addNewObstacle() { this.obstacles.push({ r: Math.random() }); } },
    keys,
    onKeyDown(e) {
      keys.push(['down', e.keyCode]);
      assert.ok(e.target, 'the game ignores events whose target is null');
    },
    onKeyUp(e) { keys.push(['up', e.keyCode]); },
  };
  return runner;
}

test('obstacle kinds follow the original types, sizes and bird heights', () => {
  assert.equal(obstacleKind(small(1)), 'cactus_small_1');
  assert.equal(obstacleKind(small(3)), 'cactus_small_3');
  assert.equal(obstacleKind(large(2)), 'cactus_large_2');
  assert.equal(obstacleKind(bird(100)), 'bird_low');
  assert.equal(obstacleKind(bird(75)), 'bird_mid');
  assert.equal(obstacleKind(bird(50)), 'bird_high');
});

test('observe converts the game state into protocol units', () => {
  const passed = { ...small(1), xPos: 0 }; // right edge 17 < dino x 23: already behind the dino
  const r = fakeRunner({ obstacles: [passed, bird(75), large(1)], yPos: 60, jumping: true });
  const obs = observe(r, 42);
  assert.equal(obs.tick, 42);
  assert.equal(obs.speed, 8.123);
  assert.deepEqual(obs.dino, { x: 23, width: 44, elev: 33, jumping: true, ducking: false });
  assert.equal(obs.obstacles.length, 2);
  assert.equal(obs.obstacles[0].kind, 'bird_mid');
  assert.equal(obs.obstacles[0].elev, 25);  // bottom at y=115, ground at y=140
  assert.equal(obs.obstacles[1].elev, 0);   // cacti stand on the ground
  assert.notEqual(obs.obstacles[0].id, obs.obstacles[1].id);
  assert.equal(observe(r, 43).obstacles[0].id, obs.obstacles[0].id, 'ids are stable across frames');
});

test('ducking dino reports the ducking width', () => {
  assert.equal(observe(fakeRunner({ ducking: true }), 1).dino.width, 59);
});

test('score is the number the game displays', () => {
  assert.equal(score(fakeRunner()), 25);
});

test('driver: jump taps space, duck holds down arrow until it expires', () => {
  const r = fakeRunner();
  const d = new Driver(r);
  d.apply('jump', 1);
  assert.deepEqual(r.keys, [['down', KEY.JUMP]]);
  r.keys.length = 0;
  d.apply('duck', 10, 5);
  assert.deepEqual(r.keys, [['down', KEY.DUCK]]);
  r.tRex.ducking = true;
  d.tick(14);
  assert.equal(r.keys.length, 1);
  d.tick(15);
  assert.deepEqual(r.keys.at(-1), ['up', KEY.DUCK]);
});

test('driver: jumping out of a duck releases the duck first', () => {
  const r = fakeRunner({ ducking: true });
  new Driver(r).apply('jump', 1);
  assert.deepEqual(r.keys, [['up', KEY.DUCK], ['down', KEY.JUMP]]);
});

test('driver: none releases a held duck and never jumps', () => {
  const r = fakeRunner({ ducking: true });
  new Driver(r).apply('none', 1);
  assert.deepEqual(r.keys, [['up', KEY.DUCK]]);
});

test('seeded obstacles: same seed, same sequence; Math.random is restored', () => {
  const draw = seed => {
    const r = fakeRunner();
    seedObstacles(r, seed);
    for (let i = 0; i < 5; i++) r.horizon.addNewObstacle(6);
    return r.horizon.obstacles.map(o => o.r);
  };
  assert.deepEqual(draw(7), draw(7));
  assert.notDeepEqual(draw(7), draw(8));
  const before = Math.random;
  draw(1);
  assert.equal(Math.random, before);
});

test('rng is deterministic and in [0, 1)', () => {
  const a = rng(3), b = rng(3);
  for (let i = 0; i < 100; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x >= 0 && x < 1);
  }
});
