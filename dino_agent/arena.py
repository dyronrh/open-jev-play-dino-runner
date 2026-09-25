"""Arena: runs the decision server in-process and plays headless episodes against it.

The brain is loaded once and stays resident. Each episode opens the real page (the original Chrome
Dino game plus web/bridge.js) in headless Chromium with ?autoplay=1, so the game, the timing and the
model's latency are exactly what a person sees in the browser.

Needs Playwright's Chromium: `playwright install chromium`, or point CHROMIUM_PATH at a Chromium binary.
"""
from __future__ import annotations

import asyncio
import os
import statistics
from collections import Counter
from pathlib import Path

from aiohttp import web

from .brains import Brain
from .policy import policy_hash
from .server import EPISODES, PolicyStore, create_app

ROOT = Path(__file__).resolve().parent.parent
# Keep background pages running at full frame rate when several episodes play at once.
CHROMIUM_ARGS = ["--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                 "--disable-backgrounding-occluded-windows", "--autoplay-policy=no-user-gesture-required"]


class Arena:
    def __init__(self, brain: Brain, policy: dict, *, max_age_ticks: int = 12, headless: bool = True):
        self.brain = brain
        self.store = PolicyStore(policy)
        self.app = create_app(brain, self.store, max_age_ticks=max_age_ticks)
        self.headless = headless
        self.runner: web.AppRunner | None = None
        self._pw = None
        self._browser = None
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
        self.url = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        if self.runner:
            await self.runner.cleanup()
            self.runner = None

    async def _chromium(self):
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            kwargs = {"headless": self.headless, "args": CHROMIUM_ARGS}
            if os.environ.get("CHROMIUM_PATH"):
                kwargs["executable_path"] = os.environ["CHROMIUM_PATH"]
            try:
                self._browser = await self._pw.chromium.launch(**kwargs)
            except Exception as e:
                raise RuntimeError("Could not start Chromium. Run `playwright install chromium` "
                                   "or set CHROMIUM_PATH to a Chromium binary.") from e
        return self._browser

    async def run_episode(self, seed: int, max_seconds: float) -> dict:
        browser = await self._chromium()
        page = await browser.new_page(viewport={"width": 800, "height": 600})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            await page.goto(f"{self.url}/?autoplay=1&seed={seed}&max_seconds={max_seconds}")
            await page.wait_for_function("window.__episode !== undefined", timeout=(max_seconds + 30) * 1000, polling=250)
            return await page.evaluate("window.__episode")
        except Exception as e:
            return {"seed": seed, "error": f"{type(e).__name__}: {e}; page errors: {errors[:3]}"}
        finally:
            await page.close()

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
