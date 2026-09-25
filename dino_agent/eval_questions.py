"""Offline perception check: does the model read the scenes correctly?

The scene vocabulary is finite (obstacle kind x distance bin x dino state), so every sentence the
policy can produce is enumerated and answered, and accuracy is exact rather than sampled. Run this
before playing: if Laya cannot tell a head-height bird from a low one here, no amount of timing
will save it in the game.

    python -m dino_agent.eval_questions --brain laya
    python -m dino_agent.eval_questions --brain laya --policy runs/<run>/best_policy.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict

from .brains import make_brain
from .policy import DINO_STATES, OBSTACLE_KINDS, QUESTION_OPTIONS, Scene, load_policy, render
from .server import add_brain_args


def scenes(policy: dict) -> list[Scene]:
    bins = policy["ttc_bins"]
    ttc_for = {"far": bins["far"] + 5, "near": (bins["far"] + bins["near"]) / 2,
               "close": (bins["near"] + bins["close"]) / 2, "touching": bins["close"] - 2}
    out = []
    for dino in DINO_STATES:
        d = {"x": 50, "width": 44, "elev": 0, "jumping": dino == "air", "ducking": dino == "duck"}
        out.append(render({"tick": 0, "speed": 10, "dino": d, "obstacles": []}, policy))
        for kind in OBSTACLE_KINDS:
            for dist, ttc in ttc_for.items():
                x = 94 + ttc * 10  # speed 10 px/tick
                obs = {"tick": 0, "speed": 10, "dino": d,
                       "obstacles": [{"id": 1, "kind": kind, "x": x, "width": 17, "height": 35, "elev": 0}]}
                s = render(obs, policy)
                assert s.truth["distance"] == dist, (kind, dist, s.ttc)
                out.append(s)
    return out


def evaluate(brain, policy: dict) -> dict:
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct: Counter = Counter()
    probs: dict[str, list] = defaultdict(list)
    latencies, mistakes = [], []
    all_scenes = scenes(policy)
    for s in all_scenes:
        t0 = time.perf_counter()
        answers = brain.answer(s, policy)
        latencies.append((time.perf_counter() - t0) * 1000)
        for qid, truth in s.truth.items():
            got = answers[qid]["choice"]
            confusion[qid][f"{truth}->{got}"] += 1
            probs[qid].append(answers[qid]["p"])
            if got == truth:
                correct[qid] += 1
            elif len(mistakes) < 15:
                mistakes.append({"text": s.text, "question": qid, "truth": truth, "answer": got, "p": answers[qid]["p"]})
    n = len(all_scenes)
    latencies.sort()
    return {
        "brain": brain.describe(),
        "checkpoint": policy["checkpoint"],
        "scenes": n,
        "accuracy": {q: round(correct[q] / n, 4) for q in QUESTION_OPTIONS},
        "mean_top_p": {q: round(statistics.mean(probs[q]), 3) for q in QUESTION_OPTIONS},
        "latency_ms": {"p50": round(latencies[n // 2], 1), "p95": round(latencies[int(n * 0.95)], 1)},
        "errors": {q: {k: v for k, v in confusion[q].items() if k.split("->")[0] != k.split("->")[1]} for q in QUESTION_OPTIONS},
        "sample_mistakes": mistakes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact perception accuracy over every scene the policy can describe.")
    add_brain_args(ap)
    args = ap.parse_args()
    policy = load_policy(args.policy)
    brain = make_brain(args.brain, device=args.device, oracle_delay_ms=args.oracle_delay_ms, checkpoint=policy["checkpoint"])
    print(json.dumps(evaluate(brain, policy), indent=2))


if __name__ == "__main__":
    main()
