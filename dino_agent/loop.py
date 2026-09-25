"""System 2: an ADK LoopAgent that improves the policy Laya plays with, between games.

    python -m dino_agent.loop --brain laya --iterations 6 --seeds 1-5 --max-seconds 120

Structure (root = SequentialAgent):

    BaselinePlayer            plays the starting policy on fixed seeds -> best_policy / best_report
    LoopAgent (max_iterations)
      Critic   (LlmAgent)     reads best_report + best_policy + feedback, proposes a JSON patch
      Player   (BaseAgent)    applies + validates the patch, plays the same seeds -> candidate_report
      Judge    (BaseAgent)    keeps the candidate only if the median score improves; escalates at target

The LLM is never in the real-time path: during a game Laya decides every tick over the WebSocket, and
the Critic only runs between games. The Judge is plain code on fixed seeds, so an LLM cannot talk
its way into a "better" policy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from .arena import Arena
from .bench import parse_seeds
from .brains import make_brain
from .policy import PolicyError, apply_patch, load_policy, policy_hash
from .server import add_brain_args

try:
    from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event, EventActions
    from google.genai import types
except ImportError as e:  # pragma: no cover - reported at startup
    raise SystemExit("The agent loop needs Google ADK: pip install -e '.[agent]'") from e

from pydantic import ConfigDict

log = logging.getLogger("dino_agent.loop")
ROOT = Path(__file__).resolve().parent.parent

# State keys shared by the agents.
BEST_POLICY, BEST_REPORT = "best_policy", "best_report"
CANDIDATE_POLICY, CANDIDATE_REPORT = "candidate_policy", "candidate_report"
PROPOSAL, FEEDBACK, HISTORY = "critic_proposal", "critic_feedback", "history"


def _event(author: str, text: str, state_delta: dict | None = None, escalate: bool = False) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]),
                 actions=EventActions(state_delta=state_delta or {}, escalate=escalate or None))


def _extract_json(text: str) -> Any:
    """Parse the first JSON object in an LLM reply, tolerating ```json fences and prose around it."""
    start = text.find("{")
    if start < 0:
        raise PolicyError("the proposal contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def headline(report: dict) -> str:
    return (f"median {report['median_score']} (min {report['min_score']}, max {report['max_score']}), "
            f"survived {report['survived']}/{report['episodes']}, crashed on {report['crashed_on'] or 'nothing'}, "
            f"rtt p95 {report['rtt_p95_ms']} ms")


class Evaluator:
    """What the player agents need: an arena plus the fixed evaluation protocol."""

    def __init__(self, arena: Arena, seeds: list[int], max_seconds: float, parallel: int = 1):
        self.arena, self.seeds, self.max_seconds, self.parallel = arena, seeds, max_seconds, parallel

    async def __call__(self, policy: dict) -> dict:
        return await self.arena.evaluate(policy, self.seeds, self.max_seconds, self.parallel)


class BaselinePlayer(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    evaluator: Any
    initial_policy: dict

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        report = await self.evaluator(self.initial_policy)
        yield _event(self.name, f"Baseline {policy_hash(self.initial_policy)}: {headline(report)}", {
            BEST_POLICY: json.dumps(self.initial_policy), BEST_REPORT: json.dumps(report),
            FEEDBACK: "", HISTORY: json.dumps([{"policy": report["policy"], "median": report["median_score"], "accepted": True}]),
        })


class Player(BaseAgent):
    """Applies the Critic's patch to the best policy and plays it. Invalid patches are not played."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    evaluator: Any

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        best = json.loads(state[BEST_POLICY])
        try:
            patch = _extract_json(str(state.get(PROPOSAL, "")))
            candidate = apply_patch(best, patch)
        except (PolicyError, ValueError) as e:
            yield _event(self.name, f"Rejected proposal before playing: {e}",
                         {CANDIDATE_POLICY: "", CANDIDATE_REPORT: json.dumps({"invalid": str(e)})})
            return
        if policy_hash(candidate) == policy_hash(best):
            yield _event(self.name, "The proposal does not change the policy.",
                         {CANDIDATE_POLICY: "", CANDIDATE_REPORT: json.dumps({"invalid": "the patch changes nothing"})})
            return
        report = await self.evaluator(candidate)
        yield _event(self.name, f"Candidate {report['policy']}: {headline(report)}",
                     {CANDIDATE_POLICY: json.dumps(candidate), CANDIDATE_REPORT: json.dumps(report)})


class Judge(BaseAgent):
    """Deterministic acceptance: the candidate must beat the best median by `min_gain` (relative)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_score: float | None = None  # None: stop when every seed survives the full episode
    min_gain: float = 0.02
    out_dir: Path

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        best_report = json.loads(state[BEST_REPORT])
        cand_report = json.loads(state.get(CANDIDATE_REPORT) or "{}")
        history = json.loads(state.get(HISTORY) or "[]")
        delta: dict = {}

        if "invalid" in cand_report or not state.get(CANDIDATE_POLICY):
            reason = cand_report.get("invalid", "no candidate")
            delta[FEEDBACK] = f"Your last proposal was rejected without playing: {reason}. Fix that and propose again."
            history.append({"policy": None, "median": None, "accepted": False, "why": reason})
            text = f"Rejected: {reason}"
        else:
            best, cand = best_report["median_score"], cand_report["median_score"]
            accepted = cand > best * (1 + self.min_gain) or (best == 0 and cand > 0)
            history.append({"policy": cand_report["policy"], "median": cand, "accepted": accepted})
            if accepted:
                delta[BEST_POLICY], delta[BEST_REPORT] = state[CANDIDATE_POLICY], state[CANDIDATE_REPORT]
                delta[FEEDBACK] = f"Your last change was ACCEPTED: median {best} -> {cand}. Build on it."
                self._save(json.loads(state[CANDIDATE_POLICY]), cand_report)
                best_report = cand_report
                text = f"Accepted {cand_report['policy']}: median {best} -> {cand}"
            else:
                delta[FEEDBACK] = (f"Your last change was REJECTED: median {cand} did not beat {best}. "
                                   f"Its report: {headline(cand_report)}. Try a different change.")
                text = f"Rejected {cand_report['policy']}: median {cand} vs best {best}"
        delta[HISTORY] = json.dumps(history)
        if self.target_score is None:
            done = best_report["survived"] == best_report["episodes"]
        else:
            done = best_report["median_score"] >= self.target_score
        if done:
            text += " - target reached, stopping."
        yield _event(self.name, text, delta, escalate=done)

    def _save(self, policy: dict, report: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "best_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
        (self.out_dir / "best_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (self.out_dir / f"policy_{report['policy']}.json").write_text(json.dumps(policy, indent=2) + "\n")


CRITIC_BRIEF = """You tune the policy of an agent that plays the Chrome-style Dino Runner game.

How the game is played, every tick, in real time:
1. Code describes the scene in English using the policy's phrases: the dino's state, the nearest
   obstacle, and a distance phrase chosen from the obstacle's time-to-contact in ticks (60 ticks per
   second) using ttc_bins: ttc >= far -> "far", >= near -> "near", >= close -> "close", else "touching".
   The time-to-contact is already reduced by the measured round-trip latency plus lead_frames.
2. Laya, a small non-generative decision model, answers two multiple-choice questions about that text
   ("obstacle": ground/head/sky/none, "distance": far/near/close/touching) with probabilities.
3. The rules map the answers to an action. A rule fires when the dino state is in its "dino" list and,
   for each question in "when", the answer is in the list with probability >= min_prob. First match
   wins; otherwise the action is "none". "duck" is held for hold_ticks unless renewed.

Game physics: a jump lasts about 35-38 ticks. Tall cacti are 50 px high, small ones 35 px, and groups
of 2-3 are wider, so they need a later jump. A pterodactyl "bird_low" must be jumped, "bird_mid" must
be ducked under, "bird_high" is harmless unless the dino is in the air. Decisions arrive about once
per round trip, so a narrow jump window can be skipped entirely when latency is high; a wide window
makes the dino jump too early. Laya does best when the phrases say plainly what is where, when each
option description is distinct, and when there are no numbers in the text.

You may change: phrases, question instructions and option descriptions (not the option keys),
ttc_bins, lead_frames, hold_ticks, min_prob, rules, checkpoint (english | multilingual). Keep
changes small and targeted at the deaths in the report: one idea per proposal.

Reply with ONE JSON object and nothing else: a patch that is deep-merged into the current policy
(objects merge key by key; lists and strings replace). Example of the shape, as text: an object with
key "ttc_bins" whose value is an object with keys "near" and "close" and numeric values.
"""


def critic_instruction(ctx) -> str:
    s = ctx.state
    return (f"{CRITIC_BRIEF}\n\nCURRENT POLICY:\n{s.get(BEST_POLICY)}\n\n"
            f"ITS REPORT (the deaths list shows what Laya was told and answered just before each crash):\n{s.get(BEST_REPORT)}\n\n"
            f"FEEDBACK ON YOUR PREVIOUS PROPOSAL:\n{s.get(FEEDBACK) or 'none yet'}\n\n"
            f"HISTORY OF PROPOSALS:\n{s.get(HISTORY)}")


def build_root(*, evaluator: Evaluator, initial_policy: dict, critic: BaseAgent, iterations: int,
               target_score: float | None, out_dir: Path, min_gain: float = 0.02) -> SequentialAgent:
    loop = LoopAgent(name="ImproveLoop", max_iterations=iterations, sub_agents=[
        critic,
        Player(name="Player", evaluator=evaluator),
        Judge(name="Judge", target_score=target_score, min_gain=min_gain, out_dir=out_dir),
    ])
    return SequentialAgent(name="DinoSystem2", sub_agents=[
        BaselinePlayer(name="BaselinePlayer", evaluator=evaluator, initial_policy=initial_policy), loop])


def make_critic(model: str) -> LlmAgent:
    if "/" in model:  # provider/model strings (anthropic/..., openai/...) go through LiteLLM
        from google.adk.models.lite_llm import LiteLlm
        model = LiteLlm(model=model)
    return LlmAgent(name="Critic", model=model, instruction=critic_instruction,
                    include_contents="none", output_key=PROPOSAL,
                    description="Proposes one targeted change to the policy as a JSON patch.")


async def run_loop(root: SequentialAgent, *, app_name: str = "dino") -> dict:
    from google.adk.runners import InMemoryRunner
    runner = InMemoryRunner(agent=root, app_name=app_name)
    session = await runner.session_service.create_session(app_name=app_name, user_id="local")
    msg = types.Content(role="user", parts=[types.Part(text="Improve the policy.")])
    async for ev in runner.run_async(user_id="local", session_id=session.id, new_message=msg):
        if ev.content and ev.content.parts and ev.author != "Critic":
            print(f"[{ev.author}] {ev.content.parts[0].text}", flush=True)
        elif ev.author == "Critic" and ev.content and ev.content.parts and ev.content.parts[0].text:
            print(f"[Critic] proposes {ev.content.parts[0].text.strip()[:400]}", flush=True)
    session = await runner.session_service.get_session(app_name=app_name, user_id="local", session_id=session.id)
    return dict(session.state)


async def main_async(args) -> None:
    policy = load_policy(args.policy)
    brain = make_brain(args.brain, device=args.device, oracle_delay_ms=args.oracle_delay_ms, checkpoint=policy["checkpoint"])
    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    async with Arena(brain, policy) as arena:
        evaluator = Evaluator(arena, parse_seeds(args.seeds), args.max_seconds, args.parallel)
        root = build_root(evaluator=evaluator, initial_policy=policy, critic=make_critic(args.model),
                          iterations=args.iterations, target_score=args.target, out_dir=out_dir, min_gain=args.min_gain)
        state = await run_loop(root)
    best = json.loads(state[BEST_REPORT])
    print(f"\nBest policy {best['policy']}: {headline(best)}")
    if (out_dir / "best_policy.json").exists():
        print(f"Saved to {out_dir}/best_policy.json")
    else:
        print("No proposal beat the starting policy.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Improve the Dino policy with an ADK LoopAgent (Critic -> Player -> Judge).")
    add_brain_args(ap)
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Critic LLM: a Gemini model id, or provider/model via LiteLLM (e.g. anthropic/claude-sonnet-5)")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--max-seconds", type=float, default=120)
    ap.add_argument("--parallel", type=int, default=1, help="episodes at once (keep 1 with a real model)")
    ap.add_argument("--target", type=float, default=None,
                    help="stop when the best median score reaches this (default: when every seed survives)")
    ap.add_argument("--min-gain", type=float, default=0.02, help="relative median improvement needed to accept")
    ap.add_argument("--out", default=str(ROOT / "runs"), help="where accepted policies are written")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
