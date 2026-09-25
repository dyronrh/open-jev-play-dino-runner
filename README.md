# open-jev-play-dino-runner

A Chrome-style Dino Runner played by **[Laya](https://github.com/NandhaKishorM/laya)**, an open-source,
non-generative decision model, over a WebSocket in near real time. An **ADK LoopAgent** improves the
policy Laya plays with, between games.

- **System 1 (fast, inside a game):** every tick, the game state is described in words. Laya answers
  two typed questions in one forward pass, and the answer becomes jump, duck or nothing.
- **System 2 (slow, between games):** a Google ADK `LoopAgent` has a Critic LLM read how the dino
  died and propose one change to the policy. It replays the same seeds and keeps the change only if
  the median score improves.

The LLM is never in the real-time loop. Laya makes every in-game decision.

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [How a decision is made](#how-a-decision-is-made)
- [The policy file](#the-policy-file)
- [The improvement loop (ADK)](#the-improvement-loop-adk)
- [WebSocket protocol](#websocket-protocol)
- [Measured so far](#measured-so-far)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)
- [Credits and licences](#credits-and-licences)

## Quick start

Requirements: Python 3.10+, Node.js 22+ (for the headless runner and the JS tests), and for Laya a
machine that can download about 2.3 GB of weights from Hugging Face. A GPU (CUDA or Apple MPS) is
strongly recommended; see [Measured so far](#measured-so-far).

```bash
git clone https://github.com/dyronrh/open-jev-play-dino-runner
cd open-jev-play-dino-runner
python -m venv .venv && source .venv/bin/activate
pip install -e '.[laya,agent,dev]'
```

**Watch Laya play** in the browser:

```bash
python -m dino_agent.server --brain laya     # first start downloads the weights, then warms up
# open http://127.0.0.1:8765
```

**Without the model:** `--brain oracle` answers from the ground truth the text was generated from.
It is not a model. It is the ceiling for a policy (perfect perception) and lets you work on
everything else without the download. `--oracle-delay-ms 35` simulates a model's latency.

```bash
python -m dino_agent.server --brain oracle --oracle-delay-ms 35
```

**Check perception offline.** Do this first with any new policy or checkpoint. The set of sentences
the policy can produce is finite, so this scores every one of them:

```bash
python -m dino_agent.eval_questions --brain laya
```

**Benchmark a policy** on fixed seeds, in real time, headless:

```bash
python -m dino_agent.bench --brain laya --seeds 1-5 --max-seconds 120
```

**Run the improvement loop** (needs a Gemini API key, or any LiteLLM model, for the Critic):

```bash
export GOOGLE_API_KEY=...
python -m dino_agent.loop --brain laya --iterations 6 --seeds 1-5 --max-seconds 120
# or: --model anthropic/claude-sonnet-5   (set ANTHROPIC_API_KEY; needs `pip install litellm`)
```

Accepted policies are written to `runs/<timestamp>/best_policy.json`. Play one with
`--policy runs/<timestamp>/best_policy.json` on any command.

**Tests:**

```bash
pytest          # policy, server protocol, headless end-to-end, ADK loop with a scripted critic
npm test        # simulation: determinism, physics, solvability, controller
```

None of the tests need the Laya weights or an API key.

## Architecture

```
┌───────────── Browser page, or node web/headless.mjs ─────────────┐
│ sim.js      deterministic game, fixed 60 Hz, seeded               │
│ agent-client.js                                                   │
│   ├─ sends the observation (numbers) when nothing is in flight    │
│   ├─ drops answers older than max_age_ticks                       │
│   └─ Controller: jump = one-tick pulse, duck = held hold_ticks    │
└──────────────────────────┬────────────────────────────────────────┘
                           │ WebSocket, JSON, 1 message in flight
┌──────────────────────────▼────────────────────────────────────────┐
│ dino_agent/server.py   (aiohttp, 127.0.0.1)                       │
│   policy.render()  numbers → English, latency-compensated         │
│   brain.answer()   Laya: 2 typed questions, 1 forward pass        │  ← one worker thread,
│   policy.decide()  answers → jump / duck / none via policy rules  │    one model, no queue
└──────────────────────────▲────────────────────────────────────────┘
                           │ policy.json swapped between games
┌──────────────────────────┴────────────────────────────────────────┐
│ dino_agent/loop.py   ADK: SequentialAgent                         │
│   BaselinePlayer → LoopAgent[ Critic (LLM) → Player → Judge ]     │
└───────────────────────────────────────────────────────────────────┘
```

Design decisions, and why:

| Decision | Why |
|---|---|
| Laya never sees numbers | Laya's integration notes show it cannot compare numbers. Code resolves distance into a time-to-contact bin and names it in words. Laya reads the words. |
| Ask what the model *sees*, not what to *do* | Perception questions ("where is it?", "how far?") are reliable for an entailment-style model. Action questions came out inverted in earlier Laya experiments. |
| One observation in flight | The server never queues, so every answer is about a fresh state. Throughput adapts to the model's latency automatically. |
| Latency compensation | The scene is described as it *will be* when the answer lands. The distance is reduced by the measured round-trip time and `lead_frames`. |
| Stale answers are dropped | An answer for a state more than `max_age_ticks` old is discarded, not applied. |
| Duck expires | Duck is held for `hold_ticks` unless renewed, so a dropped connection cannot leave the dino ducking forever. |
| Single inference worker | One GPU does one forward pass at a time. Parallel calls only interleave and inflate latency. |
| The game is always solvable | Unlike the original, every gap leaves time for a full jump plus a reaction. A death measures the decision-maker, not bad luck (`web/test/sim.test.js` proves it with an ideal bot). |
| The Judge is code | Acceptance is a deterministic median over fixed seeds. The LLM proposes; it cannot argue a policy into being accepted. |
| The Critic edits JSON, not code | Every patch is schema-validated (`policy.validate_policy`) before it is played. |

## How a decision is made

One tick, end to end:

1. The page sends `{tick, speed, dino, obstacles}` (numbers).
2. `policy.render` picks the nearest obstacle not yet passed and computes its time to contact in
   ticks. It subtracts the expected latency, bins the result (far / near / close / touching) and
   writes a sentence such as:

   > The dino is running on the ground. A pterodactyl flies at the height of the dino's head close in front of the dino.

3. Laya answers, in one forward pass:
   - `obstacle`: ground / head / sky / none, for example `head` with p = 0.91;
   - `distance`: far / near / close / touching, for example `close` with p = 0.84.
4. `policy.decide` walks the rules. The first rule whose conditions all hold, each with
   probability ≥ `min_prob`, wins. Here that is `duck`. If no rule holds, the action is `none`.
5. The client applies the action if it is still fresh.

Laya can only read text, so code has to put the scene into words. It does this with fixed phrases,
and no hidden heuristic decides anything. With `--brain laya` every action comes from Laya's
answers. There is no fallback player.

## The policy file

`policies/default.json` holds everything that is tunable. The Critic may change:

| Key | Meaning |
|---|---|
| `checkpoint` | `english` (calibrated, best general accuracy) or `multilingual` (~1.6× faster, uncalibrated) |
| `ttc_bins` | Time-to-contact thresholds in ticks: `far > near > close`. The "close" window is where the default policy jumps |
| `lead_frames` | Extra ticks of look-ahead on top of the measured latency |
| `hold_ticks` | How long a duck is held without renewal |
| `phrases` | The words for each dino state, each obstacle kind, each distance bin, and a clear path |
| `questions` | Laya question wording and option descriptions. The ids and option keys are fixed |
| `rules` | `{action, when: {question: [options]}, dino: [states]}`, first match wins |
| `min_prob` | Minimum probability per question for a rule to fire |

The question ids (`obstacle`, `distance`) and their option keys are fixed. Rules, the offline
evaluation and the ground truth all refer to them. Anything else outside the schema is rejected
with a message that says what to fix.

## The improvement loop (ADK)

```
SequentialAgent "DinoSystem2"
├── BaselinePlayer            plays the starting policy → best_policy, best_report
└── LoopAgent "ImproveLoop"   (max_iterations = --iterations)
    ├── Critic   LlmAgent     reads best_policy, best_report (with the text Laya saw and its
    │                         answers before each death), feedback and history → one JSON patch
    ├── Player   BaseAgent    validates the patch, plays the same seeds → candidate_report
    └── Judge    BaseAgent    accepts if median improves by --min-gain, saves it, escalates when
                              every seed survives (or --target is reached)
```

- The Critic's instruction is a function of session state (`critic_instruction`), with
  `include_contents="none"`. Its prompt stays small and never includes old conversation.
- An invalid or no-op patch is never played. The Judge rejects it and tells the Critic why.
- `LoopAgent` and `SequentialAgent` still work in ADK 2.x but are marked deprecated in favour of
  `Workflow`. Moving to it is a contained change in `build_root`.
- Real-time episodes take real time: 5 seeds × 120 s is up to 10 minutes per iteration. Keep
  `--parallel 1` with a real model, because parallel episodes share one inference worker and
  distort the latency being measured.

## WebSocket protocol

Version 1, JSON text frames, endpoint `/ws`. Only loopback hosts and origins are accepted, unless the server is started with a non-loopback `--host`.

```jsonc
// server → client, on connect
{"v":1,"type":"hello","brain":{"name":"laya",...},"policy":"6fe7fe1ad53a","max_age_ticks":12}
// client → server, at most one in flight
{"v":1,"type":"obs","tick":1234,"speed":8.2,"rtt_ms":31.4,
 "dino":{"x":50,"width":44,"elev":0,"jumping":false,"ducking":false},
 "obstacles":[{"id":17,"kind":"bird_mid","x":312,"width":46,"height":40,"elev":30}]}
// server → client
{"v":1,"type":"act","tick":1234,"action":"duck","hold_ticks":12,"text":"The dino is running…",
 "answers":{"obstacle":{"choice":"head","p":0.91},"distance":{"choice":"close","p":0.84}},
 "reason":"obstacle=head (0.91), distance=close (0.84)","ttc":9.1,"infer_ms":22.8,"latency_ms":23.4,
 "policy":"6fe7fe1ad53a"}
// client → server, when an episode ends
{"v":1,"type":"episode_end","seed":3,"score":1510,"outcome":"crash","crashed_on":"cactus_large_3",...}
// server → client, on a bad message (the connection stays open)
{"v":1,"type":"error","error":"speed must be a finite number"}
```

`GET /health` reports the brain, the policy hash and the number of recorded episodes.
`--log-decisions file.jsonl` records every observation, text, answer and action for analysis or
fine-tuning.

## Measured so far

All numbers below are for the oracle brain with a simulated delay. **They are not Laya results.**
Real Laya numbers still have to be collected with `eval_questions` and `bench`; please add them
here. Setup: 3 seeds, 130 s each, default policy.

| Simulated latency | Like | Result |
|---|---|---|
| 0 ms | – | every seed survives to top speed (median score 2312 over 150 s, 6 seeds) |
| 20 ms | Laya multilingual on a GPU | 3/3 survive to top speed |
| 35 ms | Laya English on a GPU | 3/3 survive to top speed |
| 60 ms | Laya multilingual on a CPU | 3/3 survive to top speed |
| 140 ms | Laya English on a CPU | dies early (median 73) |

At 140 ms a decision arrives only every ~8 ticks, which can skip the whole default "close" window.
Widening it (`ttc_bins` 30/18/5) lifts the 140 ms median to 1516. The same change breaks
zero-latency play, though: it jumps too early for three tall cacti. No single window suits every
latency, which is exactly the kind of trade-off the improvement loop exists to find per machine.

Latency figures for Laya come from the laya-playground benchmark on an M1 Max (warm, one call): 34 ms
English / 21 ms multilingual on the GPU, 139 / 58 ms on the CPU.

## Repository layout

| Path | What it is |
|---|---|
| `web/sim.js` | The game: deterministic, seeded, fixed 60 Hz, no DOM. Shared by browser and Node |
| `web/agent-client.js` | WebSocket client with flow control, staleness check, reconnect; the `Controller` |
| `web/main.js`, `web/index.html` | Browser page: draws the game, lets you or the server brain play |
| `web/headless.mjs` | Plays one real-time episode against the server, prints a JSON summary |
| `web/test/` | Node tests for the simulation and the controller |
| `dino_agent/policy.py` | Policy schema, validation, patching, text rendering, answers → action |
| `dino_agent/brains.py` | `LayaBrain` (the model) and `OracleBrain` (ground truth, for baselines and tests) |
| `dino_agent/server.py` | aiohttp server: page, `/ws`, `/health`, `/policy` |
| `dino_agent/arena.py` | In-process server plus headless episodes; the episode report |
| `dino_agent/bench.py` | CLI: benchmark a policy over seeds |
| `dino_agent/eval_questions.py` | CLI: exact perception accuracy over every scene a policy can describe |
| `dino_agent/loop.py` | The ADK agents and the loop CLI |
| `policies/default.json` | The starting policy |
| `tests/` | pytest suite |

## Contributing

Good first contributions:

- **Run it on real hardware with Laya** and add `eval_questions` and `bench` numbers to
  [Measured so far](#measured-so-far), with your device and checkpoint.
- **Improve perception:** try new phrasings in a policy file and measure them with `eval_questions`.
  A PR that changes the default policy should include before/after numbers.
- **Describe the second obstacle** too, so decisions can plan across back-to-back obstacles.
- **Fine-tune:** `--log-decisions` produces labelled data, because the ground truth is in every
  record. The Laya repository has a fine-tuning notebook.
- **Port the loop to ADK `Workflow`** when it supports LLM sub-agents.

Ground rules:

1. Run `pytest` and `npm test` before opening a PR. Both run without the model or an API key.
2. Keep the simulation deterministic. Any randomness goes through the seeded `rng`, and
   `sim.test.js` must keep proving the game is solvable.
3. Keep decisions in the model. Do not add code paths that pick an action without Laya's answers
   when running `--brain laya`. Baselines belong in `OracleBrain` and must be labelled as such.
4. A protocol change bumps `PROTOCOL_VERSION` on both sides (`server.py`, `agent-client.js`).
5. The policy schema is the contract with the Critic. If you extend it, update `validate_policy`,
   the Critic brief in `loop.py`, the tests and the table above.
6. Report numbers honestly: say which brain, device, checkpoint and seeds produced them.

## Credits and licences

- **Laya** is created by Nandakishor M (Convai Innovations) and released under Apache-2.0:
  [code](https://github.com/NandhaKishorM/laya), [weights](https://huggingface.co/convaiinnovations/laya).
  This repository installs it from PyPI and does not contain the model.
- The Laya integration guidance used here (perception questions, words instead of numbers, one
  model and one queue) comes from the [laya-playground](https://github.com/wdobry/laya-playground)
  integration skill.
- The game is an independent re-implementation inspired by the Chrome Dino game. No Chromium code
  or assets are included.
- Google ADK is Apache-2.0.

This repository does not have a licence file yet. Until the owner adds one, all rights are reserved
by default.
