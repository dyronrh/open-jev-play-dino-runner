"""Arena: runs the decision server in-process and plays headless episodes against it.

The brain is loaded once and stays resident; each episode is a `node web/headless.mjs` process that
plays in real time over the WebSocket, so the model's latency counts exactly as it does in the browser.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from collections import Counter
from pathlib import Path

from aiohttp import web

from .brains import Brain
from .policy import policy_hash
from .server import EPISODES, PolicyStore, create_app

ROOT = Path(__file__).resolve().parent.parent
HEADLESS = ROOT / "web" / "headless.mjs"


class Arena:
    def __init__(self, brain: Brain, policy: dict, *, max_age_ticks: int = 12, node: str = "node"):
        self.brain = brain
        self.store = PolicyStore(policy)
        self.app = create_app(brain, self.store, max_age_ticks=max_age_ticks)
        self.node = node
        self.runner: web.AppRunner | None = None
        self.url = ""

    async def __aenter__(self) -> "Arena":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(self) -> None:
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/ws"

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            self.runner = None

    async def run_episode(self, seed: int, max_seconds: float) -> dict:
        proc = await asyncio.create_subprocess_exec(
            self.node, str(HEADLESS), "--url", self.url, "--seed", str(seed), "--max-seconds", str(max_seconds),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        lines = [ln for ln in out.decode().splitlines() if ln.startswith("{")]
        if not lines:
            return {"seed": seed, "error": f"headless runner produced no result (exit {proc.returncode}): {err.decode()[-500:]}"}
        return json.loads(lines[-1])

    async def evaluate(self, policy: dict, seeds: list[int], max_seconds: float, parallel: int = 1) -> dict:
        """Play every seed with `policy` and summarise. Keep parallel=1 for a real model: episodes
        share one inference worker, so running them together inflates the latency being measured."""
        self.store.set(policy)
        sem = asyncio.Semaphore(max(1, parallel))

        async def one(seed: int) -> dict:
            async with sem:
                return await self.run_episode(seed, max_seconds)

        results = await asyncio.gather(*(one(s) for s in seeds))
        return summarize(results, policy, self.brain.describe(), max_seconds)

    @property
    def episodes(self):
        return self.app[EPISODES]


def summarize(results: list[dict], policy: dict, brain: dict, max_seconds: float) -> dict:
    ok = [r for r in results if "error" not in r]
    scores = [r["score"] for r in ok]

    def med(key: str):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.median(vals), 2) if vals else None

    deaths = [r for r in ok if r["outcome"] == "crash"]
    return {
        "brain": brain,
        "policy": policy_hash(policy),
        "episodes": len(results),
        "errors": [r["error"] for r in results if "error" in r],
        "max_seconds": max_seconds,
        "median_score": statistics.median(scores) if scores else 0,
        "mean_score": round(statistics.mean(scores), 1) if scores else 0,
        "min_score": min(scores, default=0),
        "max_score": max(scores, default=0),
        "survived": sum(r["outcome"] == "timeout" for r in ok),
        "crashed_on": dict(Counter(r["crashed_on"] for r in deaths)),
        "top_speed": max((r["speed"] for r in ok), default=0),
        "rtt_p50_ms": med("rtt_p50_ms"),
        "rtt_p95_ms": med("rtt_p95_ms"),
        "stale_decisions": sum(r.get("stale", 0) for r in ok),
        "timeouts": sum(r.get("timeouts", 0) for r in ok),
        "per_seed": [{k: r.get(k) for k in ("seed", "score", "outcome", "crashed_on", "speed")} for r in ok],
        # The last few decisions before each death: what the model was told and what it answered.
        "deaths": [{"seed": r["seed"], "crashed_on": r["crashed_on"], "speed": r["speed"],
                    "last_decisions": r.get("last_decisions", [])[-3:]} for r in deaths[:4]],
    }
