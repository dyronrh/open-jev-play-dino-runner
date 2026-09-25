// Environment adapter for the original Chrome Dino game (web/original/index.js).
//
// The game is not modified. Everything here goes through its public global, `Runner.instance_`:
//   - observe():  read the T-Rex and obstacle state into plain numbers for the decision server
//   - Driver:     press and release keys by calling the game's own key handlers, exactly as a
//                 keyboard would, so the game's rules (no jump while ducking, speed drop in the
//                 air, etc.) apply unchanged
//   - seedObstacles(): make the obstacle sequence reproducible per seed
//
// Pure functions of a runner object, so they are unit-tested in Node with a fake runner.

export const KEY = { JUMP: 32, DUCK: 40 };

// Original geometry (desktop): the ground line is at y = HEIGHT - BOTTOM_PAD = 140.
const GROUND_Y = 140;
const BIRD_LEVEL = { 100: 'low', 75: 'mid', 50: 'high' };

// mulberry32: tiny, fast, seedable PRNG.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function obstacleKind(o) {
  const type = o.typeConfig.type;
  if (type === 'CACTUS_SMALL') return `cactus_small_${o.size}`;
  if (type === 'CACTUS_LARGE') return `cactus_large_${o.size}`;
  if (type === 'PTERODACTYL') return `bird_${BIRD_LEVEL[o.yPos] || (o.yPos >= 90 ? 'low' : o.yPos >= 65 ? 'mid' : 'high')}`;
  return 'unknown';
}

// Stable ids for obstacles, which the game does not number itself.
const ids = new WeakMap();
let nextId = 1;
export function obstacleId(o) {
  if (!ids.has(o)) ids.set(o, nextId++);
  return ids.get(o);
}

export function score(runner) {
  return runner.distanceMeter.getActualDistance(Math.ceil(runner.distanceRan));
}

// Observation in the protocol's units: px, and speed in px per 60 Hz frame (the game's own unit).
export function observe(runner, tick) {
  const t = runner.tRex;
  const width = t.ducking ? t.config.WIDTH_DUCK : t.config.WIDTH;
  return {
    tick,
    speed: +runner.currentSpeed.toFixed(3),
    dino: { x: t.xPos, width, elev: +(t.groundYPos - t.yPos).toFixed(2), jumping: !!t.jumping, ducking: !!t.ducking },
    obstacles: runner.horizon.obstacles
      .filter(o => o.xPos + o.width > t.xPos) // not yet passed
      .slice(0, 2)
      .map(o => ({
        id: obstacleId(o), kind: obstacleKind(o), x: +o.xPos.toFixed(2), width: o.width,
        height: o.typeConfig.height, elev: GROUND_Y - (o.yPos + o.typeConfig.height),
      })),
  };
}

// Presses keys through the game's own handlers. Jump is a tap (like a bot pressing space);
// duck is held until `hold` ticks pass without renewal, then released.
export class Driver {
  constructor(runner) {
    this.runner = runner;
    this.duckUntil = -1;
  }

  key(type, keyCode) {
    // `target` must be non-null: the game ignores keys whose target is its (absent) details button.
    const e = { type, keyCode, target: {}, currentTarget: null, preventDefault() {} };
    if (type === 'keydown') this.runner.onKeyDown(e);
    else this.runner.onKeyUp(e);
  }

  apply(action, tick, hold = 12) {
    const t = this.runner.tRex;
    if (action === 'jump') {
      if (t.ducking) this.releaseDuck(); // the game ignores jump while ducking
      this.duckUntil = -1;
      if (!t.jumping) this.key('keydown', KEY.JUMP);
    } else if (action === 'duck') {
      this.duckUntil = tick + hold;
      if (!t.ducking) this.key('keydown', KEY.DUCK);
    } else if (t.ducking) {
      this.releaseDuck();
    }
  }

  releaseDuck() {
    this.duckUntil = -1;
    this.key('keyup', KEY.DUCK);
  }

  // Call once per tick: releases an expired duck.
  tick(tick) {
    if (this.duckUntil >= 0 && tick >= this.duckUntil) this.releaseDuck();
  }
}

// Make obstacles reproducible: while the game creates an obstacle (type, size, bird height, gap),
// Math.random is swapped for a PRNG seeded per episode. Clouds, blinking etc. stay random.
export function seedObstacles(runner, seed) {
  const horizon = runner.horizon;
  horizon.__rand = rng(seed);
  if (!horizon.__seeded) {
    const original = horizon.addNewObstacle;
    horizon.addNewObstacle = function (...args) {
      const saved = Math.random;
      Math.random = this.__rand;
      try { return original.apply(this, args); } finally { Math.random = saved; }
    };
    horizon.__seeded = true;
  }
}
