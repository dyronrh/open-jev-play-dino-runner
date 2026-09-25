// Deterministic Dino Runner simulation. Pure logic, no DOM: the browser page and the headless
// Node runner both drive this same file, so an episode with a given seed plays identically in both.
//
// Units: pixels and ticks. One tick is one frame at 60 Hz. Heights are measured upwards from the
// ground ("elev" = how far the bottom of a sprite is above the ground).

export const TICK_HZ = 60;
export const WIDTH = 600;
export const HEIGHT = 150;
export const GROUND_Y = 130; // canvas y of the ground line, used by the renderer

export const DINO = { x: 50, width: 44, height: 47, duckWidth: 59, duckHeight: 25 };
export const PHYSICS = {
  gravity: 0.6,
  jumpVelocity: 10,       // plus speed / 10, as in the original game
  fastFallFactor: 3,      // gravity multiplier while ducking in the air
  startSpeed: 6,
  maxSpeed: 13,
  acceleration: 0.001,    // speed gained per tick
  clearTicks: 180,        // no obstacles during the first 3 seconds
  hitboxInset: 3,         // forgiving collisions, px trimmed from every side
  reactionTicks: 12,      // every gap leaves a full jump plus this long to react (see newObstacle)
};

// Obstacle catalogue. `multipleSpeed`: above this speed a cactus can come in groups of 2 or 3.
export const OBSTACLE_TYPES = {
  cactus_small: { width: 17, height: 35, minGap: 120, minSpeed: 0, multipleSpeed: 4 },
  cactus_large: { width: 25, height: 50, minGap: 120, minSpeed: 0, multipleSpeed: 7 },
  bird: { width: 46, height: 40, minGap: 150, minSpeed: 8.5, multipleSpeed: Infinity },
};
// Bird flight levels: low must be jumped, mid must be ducked under, high is harmless on the ground.
export const BIRD_ELEVATIONS = { low: 12, mid: 30, high: 50 };
const GAP_COEFFICIENT = 0.6;
const MAX_GAP_COEFFICIENT = 1.5;
const MAX_DUPLICATION = 2;

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

export class DinoSim {
  constructor(seed = 1) {
    this.reset(seed);
  }

  reset(seed = this.seed) {
    this.seed = seed;
    this.rand = rng(seed);
    this.tick = 0;
    this.speed = PHYSICS.startSpeed;
    this.distance = 0;
    this.crashed = null; // set to the obstacle that ended the run
    this.dino = { elev: 0, vy: 0, jumping: false, ducking: false };
    this.obstacles = [];
    this.nextId = 1;
    this.history = []; // kinds of recent obstacles, to limit repeats
  }

  get score() {
    return Math.floor(this.distance * 0.025);
  }

  // Advance one tick. `input` = { jump: bool, duck: bool }; duck is level-triggered (held).
  step(input = {}) {
    if (this.crashed) return;
    this.tick++;
    this.updateDino(input);
    this.speed = Math.min(PHYSICS.maxSpeed, this.speed + PHYSICS.acceleration);
    this.distance += this.speed;
    for (const o of this.obstacles) o.x -= this.speed;
    this.obstacles = this.obstacles.filter(o => o.x + o.width > -10);
    this.maybeSpawn();
    const hit = this.obstacles.find(o => this.collides(o));
    if (hit) this.crashed = { id: hit.id, kind: hit.kind, tick: this.tick };
  }

  updateDino({ jump = false, duck = false }) {
    const d = this.dino;
    if (jump && !d.jumping) {
      d.jumping = true;
      d.vy = PHYSICS.jumpVelocity + this.speed / 10;
    }
    if (d.jumping) {
      const g = duck ? PHYSICS.gravity * PHYSICS.fastFallFactor : PHYSICS.gravity;
      d.elev += d.vy;
      d.vy -= g;
      if (d.elev <= 0) {
        d.elev = 0;
        d.vy = 0;
        d.jumping = false;
      }
    }
    d.ducking = duck && !d.jumping;
  }

  dinoBox() {
    const d = this.dino, i = PHYSICS.hitboxInset;
    const w = d.ducking ? DINO.duckWidth : DINO.width;
    const h = d.ducking ? DINO.duckHeight : DINO.height;
    return { x0: DINO.x + i, x1: DINO.x + w - i, y0: d.elev + i, y1: d.elev + h - i };
  }

  collides(o) {
    const a = this.dinoBox(), i = PHYSICS.hitboxInset;
    return a.x0 < o.x + o.width - i && a.x1 > o.x + i && a.y0 < o.elev + o.height - i && a.y1 > o.elev + i;
  }

  maybeSpawn() {
    if (this.tick < PHYSICS.clearTicks) return;
    const last = this.obstacles[this.obstacles.length - 1];
    if (last && last.x + last.width + last.gap > WIDTH) return;
    this.obstacles.push(this.newObstacle());
  }

  newObstacle() {
    const choices = Object.keys(OBSTACLE_TYPES).filter(t => {
      if (this.speed < OBSTACLE_TYPES[t].minSpeed) return false;
      const recent = this.history.slice(-MAX_DUPLICATION);
      return !(recent.length === MAX_DUPLICATION && recent.every(k => k === t));
    });
    const type = choices[Math.floor(this.rand() * choices.length)];
    const spec = OBSTACLE_TYPES[type];
    const size = this.speed > spec.multipleSpeed ? 1 + Math.floor(this.rand() * 3) : 1;
    const width = spec.width * size;
    let kind = `${type}_${size}`, elev = 0;
    if (type === 'bird') {
      const level = ['low', 'mid', 'high'][Math.floor(this.rand() * 3)];
      kind = `bird_${level}`;
      elev = BIRD_ELEVATIONS[level];
    }
    const minGap = Math.round(width * this.speed + spec.minGap * GAP_COEFFICIENT);
    let gap = Math.round(minGap + this.rand() * (minGap * MAX_GAP_COEFFICIENT - minGap));
    // Unlike the original, never spawn a gap the dino cannot land and react in: the game is always
    // solvable, so a death measures the decision maker rather than bad luck.
    const airTicks = (2 * (PHYSICS.jumpVelocity + this.speed / 10)) / PHYSICS.gravity;
    gap = Math.max(gap, Math.ceil(this.speed * (airTicks + PHYSICS.reactionTicks)));
    this.history.push(type);
    if (this.history.length > 8) this.history.shift();
    return { id: this.nextId++, kind, x: WIDTH, width, height: spec.height, elev, gap };
  }

  // Structured observation sent to the decision server. Plain numbers; the server turns them into words.
  observe() {
    const d = this.dino;
    return {
      tick: this.tick,
      speed: +this.speed.toFixed(3),
      dino: { x: DINO.x, width: d.ducking ? DINO.duckWidth : DINO.width, elev: +d.elev.toFixed(2), jumping: d.jumping, ducking: d.ducking },
      obstacles: this.obstacles
        .filter(o => o.x + o.width > DINO.x) // not yet passed
        .slice(0, 2)
        .map(o => ({ id: o.id, kind: o.kind, x: +o.x.toFixed(2), width: o.width, height: o.height, elev: o.elev })),
    };
  }
}
