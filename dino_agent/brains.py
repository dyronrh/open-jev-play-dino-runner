"""Brains answer the policy's questions about a scene.

- LayaBrain: the real thing. Laya reads the scene text and answers both questions in one forward pass.
- OracleBrain: answers from the ground truth the text was generated from, optionally after a fixed
  delay. It is the ceiling for a given policy (perfect perception) and lets you test timing and the
  whole pipeline without downloading the model. It is never the model, and reports say which brain ran.

All brains are synchronous and are called from a single worker thread, one call at a time.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .policy import Scene, decide, render


@dataclass
class Decision:
    action: str
    reason: str
    text: str
    answers: dict        # {qid: {"choice": str, "p": float}}
    target_id: int | None
    ttc: float | None
    truth: dict
    infer_ms: float


class Brain:
    name = "brain"

    def answer(self, scene: Scene, policy: dict) -> dict:
        raise NotImplementedError

    def decide(self, obs: dict, policy: dict, latency_ms: float = 0.0) -> Decision:
        scene = render(obs, policy, latency_ms)
        t0 = time.perf_counter()
        answers = self.answer(scene, policy)
        infer_ms = (time.perf_counter() - t0) * 1000
        action, reason = decide(answers, scene, policy)
        return Decision(action, reason, scene.text, answers, scene.target_id, scene.ttc, scene.truth, round(infer_ms, 2))

    def describe(self) -> dict:
        return {"name": self.name}


class OracleBrain(Brain):
    """Perfect perception from the ground truth. Not a model: a baseline and a test double."""
    name = "oracle"

    def __init__(self, delay_ms: float = 0.0):
        self.delay_ms = delay_ms

    def answer(self, scene: Scene, policy: dict) -> dict:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)
        return {qid: {"choice": choice, "p": 1.0} for qid, choice in scene.truth.items()}

    def describe(self) -> dict:
        return {"name": self.name, "delay_ms": self.delay_ms}


class LayaBrain(Brain):
    """Laya (https://github.com/NandhaKishorM/laya): typed questions in, probabilities out."""
    name = "laya"

    def __init__(self, device: str | None = None, preload: str | None = "english"):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import laya  # imported lazily: the oracle brain and the tests must not need it

        self._laya = laya
        self.device = device
        self.agents: dict[str, object] = {}
        if preload:
            self._agent(preload)

    def _agent(self, checkpoint: str):
        if checkpoint not in self.agents:
            sub = None if checkpoint == "english" else checkpoint
            kwargs = {"subfolder": sub} if sub else {}
            if self.device:
                kwargs["device"] = self.device
            agent = self._laya.load("convaiinnovations/laya", **kwargs)
            self.agents[checkpoint] = agent
            self._warm_up(agent)
        return self.agents[checkpoint]

    @staticmethod
    def _warm_up(agent) -> None:
        # The first calls at a new shape compile kernels and are several times slower.
        q = {"x": {"type": "choice", "instructions": "Is it far?", "criteria": {"a": "far", "b": "near"}}}
        for _ in range(3):
            agent.predict("A cactus stands on the ground far ahead of the dino.", q)

    def answer(self, scene: Scene, policy: dict) -> dict:
        agent = self._agent(policy["checkpoint"])
        result = agent.predict(scene.text, policy["questions"])
        out = {}
        for qid, a in result["answers"].items():
            probs = a.get("probabilities") or {}
            out[qid] = {"choice": a["choice"], "p": round(float(probs.get(a["choice"], 0.0)), 4)}
        return out

    def describe(self) -> dict:
        devices = {k: str(getattr(a, "device", "?")) for k, a in self.agents.items()}
        return {"name": self.name, "laya": getattr(self._laya, "__version__", "?"), "loaded": devices}


def make_brain(name: str, *, device: str | None = None, oracle_delay_ms: float = 0.0, checkpoint: str = "english") -> Brain:
    if name == "laya":
        return LayaBrain(device=device, preload=checkpoint)
    if name == "oracle":
        return OracleBrain(delay_ms=oracle_delay_ms)
    raise ValueError(f"unknown brain {name!r} (expected laya or oracle)")
