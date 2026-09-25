"""The decision policy: how a game observation becomes words for the model, and how the model's
answers become an action.

A policy is plain JSON (see policies/default.json) so that the Critic agent can rewrite it without
touching code. Everything here is deterministic and cheap; the model call happens in brains.py.

What the code does and does not decide
--------------------------------------
Laya reads text, not pixels or numbers, so code must describe the scene in words. It does that with
fixed phrases per obstacle kind and per time-to-contact bin. The *decision* comes from the model's
answers to two perception questions ("where is the thing ahead?", "how far is it?"); `decide` only
maps those answers to jump / duck / none through the policy's rules.

The question ids and their option keys are fixed (they are what the rules and the offline
evaluation refer to). Their wording, the phrases, the bins and the thresholds are all tunable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TICK_HZ = 60
OBSTACLE_KINDS = (
    "cactus_small_1", "cactus_small_2", "cactus_small_3",
    "cactus_large_1", "cactus_large_2", "cactus_large_3",
    "bird_low", "bird_mid", "bird_high",
)
# Ground truth for the `obstacle` question, per kind.
OBSTACLE_CLASS = {k: "ground" for k in OBSTACLE_KINDS} | {"bird_mid": "head", "bird_high": "sky"}
QUESTION_OPTIONS = {
    "obstacle": ("ground", "head", "sky", "none"),
    "distance": ("far", "near", "close", "touching"),
}
DINO_STATES = ("ground", "air", "duck")
ACTIONS = ("jump", "duck", "none")
CHECKPOINTS = ("english", "multilingual", "typed-decisions")
TOP_LEVEL_KEYS = {"version", "checkpoint", "lead_frames", "hold_ticks", "ttc_bins", "phrases", "questions", "rules", "min_prob"}
MAX_PHRASE_CHARS = 200


class PolicyError(ValueError):
    """The policy (or a patch to it) is invalid. The message says what to fix."""


def load_policy(path: str | Path) -> dict:
    policy = json.loads(Path(path).read_text())
    validate_policy(policy)
    return policy


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PolicyError(msg)


def _phrase(value: Any, where: str) -> None:
    _require(isinstance(value, str) and 0 < len(value) <= MAX_PHRASE_CHARS,
             f"{where} must be a non-empty string of at most {MAX_PHRASE_CHARS} characters")


def validate_policy(p: dict) -> None:
    _require(isinstance(p, dict), "policy must be a JSON object")
    unknown = set(p) - TOP_LEVEL_KEYS
    _require(not unknown, f"unknown top-level keys: {sorted(unknown)}")
    missing = TOP_LEVEL_KEYS - set(p)
    _require(not missing, f"missing top-level keys: {sorted(missing)}")
    _require(p["version"] == 1, "version must be 1")
    _require(p["checkpoint"] in CHECKPOINTS, f"checkpoint must be one of {CHECKPOINTS}")
    _require(isinstance(p["lead_frames"], (int, float)) and -10 <= p["lead_frames"] <= 20, "lead_frames must be a number in [-10, 20]")
    _require(isinstance(p["hold_ticks"], int) and 1 <= p["hold_ticks"] <= 60, "hold_ticks must be an integer in [1, 60]")

    bins = p["ttc_bins"]
    _require(isinstance(bins, dict) and set(bins) == {"far", "near", "close"}, "ttc_bins needs exactly far, near, close")
    _require(all(isinstance(v, (int, float)) for v in bins.values()), "ttc_bins values must be numbers (ticks)")
    _require(bins["far"] > bins["near"] > bins["close"] >= -5, "ttc_bins must satisfy far > near > close >= -5")

    ph = p["phrases"]
    _require(isinstance(ph, dict) and set(ph) == {"dino", "obstacle", "distance", "clear"}, "phrases needs dino, obstacle, distance, clear")
    for group, keys in (("dino", DINO_STATES), ("obstacle", OBSTACLE_KINDS), ("distance", QUESTION_OPTIONS["distance"])):
        _require(isinstance(ph[group], dict) and set(ph[group]) == set(keys), f"phrases.{group} needs exactly {list(keys)}")
        for k, v in ph[group].items():
            _phrase(v, f"phrases.{group}.{k}")
    _phrase(ph["clear"], "phrases.clear")

    qs = p["questions"]
    _require(isinstance(qs, dict) and set(qs) == set(QUESTION_OPTIONS), f"questions needs exactly {list(QUESTION_OPTIONS)}")
    for qid, q in qs.items():
        _require(isinstance(q, dict) and q.get("type") == "choice", f"questions.{qid}.type must be 'choice'")
        _phrase(q.get("instructions"), f"questions.{qid}.instructions")
        crit = q.get("criteria")
        _require(isinstance(crit, dict) and set(crit) == set(QUESTION_OPTIONS[qid]),
                 f"questions.{qid}.criteria needs exactly the options {list(QUESTION_OPTIONS[qid])}")
        for k, v in crit.items():
            _phrase(v, f"questions.{qid}.criteria.{k}")

    rules = p["rules"]
    _require(isinstance(rules, list) and 1 <= len(rules) <= 8, "rules must be a list of 1 to 8 rules")
    for i, r in enumerate(rules):
        _require(isinstance(r, dict) and set(r) == {"action", "when", "dino"}, f"rules[{i}] needs exactly action, when, dino")
        _require(r["action"] in ("jump", "duck"), f"rules[{i}].action must be jump or duck")
        _require(isinstance(r["dino"], list) and r["dino"] and set(r["dino"]) <= set(DINO_STATES), f"rules[{i}].dino must be a subset of {DINO_STATES}")
        _require(isinstance(r["when"], dict) and r["when"], f"rules[{i}].when must be a non-empty object")
        for qid, opts in r["when"].items():
            _require(qid in QUESTION_OPTIONS, f"rules[{i}].when refers to unknown question {qid!r}")
            _require(isinstance(opts, list) and opts and set(opts) <= set(QUESTION_OPTIONS[qid]),
                     f"rules[{i}].when.{qid} must be a non-empty subset of {QUESTION_OPTIONS[qid]}")

    mp = p["min_prob"]
    _require(isinstance(mp, dict) and set(mp) == set(QUESTION_OPTIONS), f"min_prob needs exactly {list(QUESTION_OPTIONS)}")
    _require(all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in mp.values()), "min_prob values must be in [0, 1]")


def apply_patch(policy: dict, patch: dict) -> dict:
    """Deep-merge `patch` into a copy of `policy` (objects merge, everything else replaces), then validate."""
    _require(isinstance(patch, dict), "patch must be a JSON object")

    def merge(base: Any, upd: Any) -> Any:
        if isinstance(base, dict) and isinstance(upd, dict):
            out = dict(base)
            for k, v in upd.items():
                out[k] = merge(base.get(k), v) if k in base else copy.deepcopy(v)
            return out
        return copy.deepcopy(upd)

    new = merge(copy.deepcopy(policy), patch)
    validate_policy(new)
    return new


@dataclass
class Scene:
    """One observation put into words, plus the ground truth the words were generated from."""
    text: str
    dino: str
    target_id: int | None
    target_kind: str | None
    ttc: float | None  # latency-compensated time to contact, in ticks
    truth: dict[str, str] = field(default_factory=dict)


def distance_bin(ttc: float, bins: dict) -> str:
    if ttc >= bins["far"]:
        return "far"
    if ttc >= bins["near"]:
        return "near"
    if ttc >= bins["close"]:
        return "close"
    return "touching"


def render(obs: dict, policy: dict, latency_ms: float = 0.0) -> Scene:
    """Describe the observation in words. `latency_ms` is the expected delay until the answer is
    applied; the scene is described as it will be by then, not as it is now."""
    d = obs["dino"]
    dino = "air" if d.get("jumping") else "duck" if d.get("ducking") else "ground"
    ph = policy["phrases"]
    obstacles = obs.get("obstacles") or []
    if not obstacles:
        return Scene(f'{ph["dino"][dino]} {ph["clear"]}', dino, None, None, None,
                     {"obstacle": "none", "distance": "far"})
    o = obstacles[0]
    speed = max(float(obs["speed"]), 0.1)
    gap = float(o["x"]) - (float(d["x"]) + float(d["width"]))
    ttc = gap / speed - (latency_ms * TICK_HZ / 1000.0 + policy["lead_frames"])
    kind = o["kind"] if o["kind"] in OBSTACLE_KINDS else "cactus_small_1"
    bin_ = distance_bin(ttc, policy["ttc_bins"])
    text = f'{ph["dino"][dino]} {ph["obstacle"][kind]} {ph["distance"][bin_]}'
    return Scene(text, dino, o["id"], kind, round(ttc, 2), {"obstacle": OBSTACLE_CLASS[kind], "distance": bin_})


def decide(answers: dict, scene: Scene, policy: dict) -> tuple[str, str]:
    """Map normalised answers ({qid: {"choice", "p"}}) to an action through the policy rules."""
    for rule in policy["rules"]:
        if scene.dino not in rule["dino"]:
            continue
        ok = True
        for qid, allowed in rule["when"].items():
            a = answers.get(qid)
            if not a or a["choice"] not in allowed or a["p"] < policy["min_prob"][qid]:
                ok = False
                break
        if ok:
            why = ", ".join(f'{q}={answers[q]["choice"]} ({answers[q]["p"]:.2f})' for q in rule["when"])
            return rule["action"], why
    return "none", "no rule matched"
