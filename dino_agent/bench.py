"""Benchmark a policy: play a set of seeds and print the summary.

    python -m dino_agent.bench --brain laya --seeds 1-5 --max-seconds 120
    python -m dino_agent.bench --brain oracle --seeds 1-10 --parallel 10        # ceiling, no model
    python -m dino_agent.bench --brain oracle --oracle-delay-ms 35 --seeds 1-5   # what 35 ms of latency costs
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .arena import Arena
from .brains import make_brain
from .policy import load_policy
from .server import add_brain_args


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


async def run(args) -> dict:
    policy = load_policy(args.policy)
    brain = make_brain(args.brain, device=args.device, oracle_delay_ms=args.oracle_delay_ms, checkpoint=policy["checkpoint"])
    async with Arena(brain, policy) as arena:
        return await arena.evaluate(policy, parse_seeds(args.seeds), args.max_seconds, args.parallel)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a policy over several seeded episodes.")
    add_brain_args(ap)
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1,4,9")
    ap.add_argument("--max-seconds", type=float, default=120)
    ap.add_argument("--parallel", type=int, default=1, help="episodes at once (keep 1 for a real model)")
    ap.add_argument("--full", action="store_true", help="print the full report including the deaths")
    args = ap.parse_args()
    report = asyncio.run(run(args))
    if not args.full:
        report.pop("deaths")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
