// Headless episode runner: plays one real-time episode against the decision server and prints a
// JSON summary on the last line of stdout. Used by the ADK PlayerAgent and by the tests.
//
//   node web/headless.mjs --url ws://127.0.0.1:8765/ws --seed 7 --max-seconds 120
//
// Real time on purpose: the model's latency is part of what is being measured.
import { DinoSim, TICK_HZ } from './sim.js';
import { AgentClient, Controller, percentile } from './agent-client.js';

function parseArgs(argv) {
  const args = { url: 'ws://127.0.0.1:8765/ws', seed: 1, maxSeconds: 120, keep: 5 };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    args[key] = key === 'url' ? argv[i + 1] : Number(argv[i + 1]);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  const sim = new DinoSim(args.seed);
  const controller = new Controller();
  const recent = []; // last decisions, reported when the dino dies
  const client = new AgentClient(args.url, {
    reconnect: false,
    onAct: msg => {
      if (!client.fresh(msg, sim.tick)) return;
      controller.apply(msg.action, sim.tick, msg.hold_ticks);
      recent.push({ tick: msg.tick, applied_at: sim.tick, action: msg.action, text: msg.text, answers: msg.answers, latency_ms: msg.latency_ms });
      if (recent.length > args.keep) recent.shift();
    },
  });
  if (!(await client.connect())) {
    console.log(JSON.stringify({ seed: args.seed, error: `could not connect to ${args.url}` }));
    process.exit(2);
  }

  const dt = 1000 / TICK_HZ, maxTicks = args.maxSeconds * TICK_HZ;
  let next = performance.now();
  await new Promise(resolve => {
    const loop = () => {
      let steps = 0;
      while (performance.now() >= next && steps < 5) { // cap catch-up after a stall
        sim.step(controller.input(sim.tick));
        client.maybeSend(sim.observe());
        next += dt;
        steps++;
        if (sim.crashed || sim.tick >= maxTicks || !client.ready) return resolve();
      }
      if (steps === 5) next = performance.now(); // fell behind: resync instead of fast-forwarding
      setTimeout(loop, Math.max(0, next - performance.now()));
    };
    loop();
  });

  const s = client.stats;
  const summary = {
    seed: args.seed,
    score: sim.score,
    ticks: sim.tick,
    seconds: +(sim.tick / TICK_HZ).toFixed(2),
    speed: +sim.speed.toFixed(2),
    outcome: sim.crashed ? 'crash' : sim.tick >= maxTicks ? 'timeout' : 'disconnected',
    crashed_on: sim.crashed ? sim.crashed.kind : null,
    decisions: s.applied,
    stale: s.stale,
    timeouts: s.timeouts,
    rtt_p50_ms: percentile(s.rtts, 0.5),
    rtt_p95_ms: percentile(s.rtts, 0.95),
    rtt_p99_ms: percentile(s.rtts, 0.99),
    last_decisions: sim.crashed ? recent : [],
  };
  client.sendEpisodeEnd(summary);
  await new Promise(r => setTimeout(r, 50)); // let the last frame flush
  client.close();
  console.log(JSON.stringify(summary));
}

main().catch(err => {
  console.log(JSON.stringify({ error: String(err && err.stack || err) }));
  process.exit(1);
});
