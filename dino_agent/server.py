"""Decision server: serves the game page and answers observations over a WebSocket.

    python -m dino_agent.server --brain laya          # then open http://127.0.0.1:8765
    python -m dino_agent.server --brain oracle        # no model, perfect perception (baseline)

Binds to loopback by default. Protocol (JSON, version 1):

    server -> client  {"v":1,"type":"hello","brain":{...},"policy":"<hash>","max_age_ticks":12}
    client -> server  {"v":1,"type":"obs","tick":..,"speed":..,"dino":{..},"obstacles":[..],"rtt_ms":..}
    server -> client  {"v":1,"type":"act","tick":..,"action":"jump|duck|none","hold_ticks":..,
                       "text":"..","answers":{..},"reason":"..","latency_ms":..,"policy":"<hash>"}
    client -> server  {"v":1,"type":"episode_end","seed":..,"score":..,...}
    server -> client  {"v":1,"type":"error","error":".."}

The client keeps at most one observation in flight, so the server never queues.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from aiohttp import WSMsgType, web

from .brains import Brain, make_brain
from .policy import load_policy, policy_hash, validate_policy

log = logging.getLogger("dino_agent.server")
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DEFAULT_POLICY = ROOT / "policies" / "default.json"
PROTOCOL_VERSION = 1
MAX_MSG_BYTES = 64 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

BRAIN = web.AppKey("brain", Brain)
POLICY = web.AppKey("policy", object)
EPISODES = web.AppKey("episodes", deque)
EXECUTOR = web.AppKey("executor", ThreadPoolExecutor)
SETTINGS = web.AppKey("settings", dict)


class PolicyStore:
    """The live policy. Swapped atomically; every decision reads it once."""

    def __init__(self, policy: dict):
        self.set(policy)

    def set(self, policy: dict) -> None:
        validate_policy(policy)
        self.policy, self.hash = policy, policy_hash(policy)


def _hostname(hostport: str) -> str:
    if hostport.startswith("["):  # [::1]:8765
        return hostport[: hostport.find("]") + 1]
    return hostport.rsplit(":", 1)[0]


def _host_ok(request: web.Request) -> bool:
    # DNS-rebinding guard: only answer requests addressed to a loopback name.
    return _hostname(request.host or "") in LOCAL_HOSTS or request.app[SETTINGS]["allow_any_host"]


def _origin_ok(request: web.Request) -> bool:
    origin = request.headers.get("Origin")
    if origin is None:  # non-browser clients (the headless runner) send no Origin
        return True
    return _hostname(origin.split("://", 1)[-1]) in LOCAL_HOSTS or request.app[SETTINGS]["allow_any_host"]


def _valid_obs(msg: dict) -> str | None:
    if not isinstance(msg.get("tick"), int):
        return "tick must be an integer"
    if not isinstance(msg.get("speed"), (int, float)) or not math.isfinite(msg["speed"]):
        return "speed must be a finite number"
    d = msg.get("dino")
    if not isinstance(d, dict) or not all(isinstance(d.get(k), (int, float)) for k in ("x", "width", "elev")):
        return "dino must have numeric x, width, elev"
    obs = msg.get("obstacles")
    if not isinstance(obs, list) or len(obs) > 5:
        return "obstacles must be a list of at most 5"
    for o in obs:
        if not isinstance(o, dict) or not isinstance(o.get("kind"), str) or not all(isinstance(o.get(k), (int, float)) for k in ("id", "x", "width")):
            return "each obstacle needs id, kind, x, width"
    return None


async def index(request: web.Request) -> web.StreamResponse:
    if not _host_ok(request):
        raise web.HTTPForbidden(text="forbidden host")
    return web.FileResponse(WEB_DIR / "index.html")


async def health(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response({"ok": True, "protocol": PROTOCOL_VERSION, "brain": app[BRAIN].describe(),
                              "policy": app[POLICY].hash, "episodes": len(app[EPISODES])})


async def get_policy(request: web.Request) -> web.Response:
    if not _host_ok(request):
        raise web.HTTPForbidden(text="forbidden host")
    return web.json_response(request.app[POLICY].policy)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    if not (_host_ok(request) and _origin_ok(request)):
        raise web.HTTPForbidden(text="forbidden origin")
    app = request.app
    ws = web.WebSocketResponse(max_msg_size=MAX_MSG_BYTES, heartbeat=10)
    await ws.prepare(request)
    settings = app[SETTINGS]
    await ws.send_json({"v": PROTOCOL_VERSION, "type": "hello", "brain": app[BRAIN].describe(),
                        "policy": app[POLICY].hash, "max_age_ticks": settings["max_age_ticks"]})
    latency_ema: float | None = None
    loop = asyncio.get_running_loop()
    decision_log = settings["decision_log"]

    async for raw in ws:
        if raw.type != WSMsgType.TEXT:
            if raw.type == WSMsgType.ERROR:
                log.warning("websocket error: %s", ws.exception())
            continue
        try:
            msg = json.loads(raw.data)
        except json.JSONDecodeError:
            await ws.send_json({"v": PROTOCOL_VERSION, "type": "error", "error": "invalid JSON"})
            continue
        if msg.get("v") != PROTOCOL_VERSION:
            await ws.send_json({"v": PROTOCOL_VERSION, "type": "error", "error": f"protocol version must be {PROTOCOL_VERSION}"})
            continue

        if msg.get("type") == "obs":
            err = _valid_obs(msg)
            if err:
                await ws.send_json({"v": PROTOCOL_VERSION, "type": "error", "error": err, "tick": msg.get("tick")})
                continue
            rtt = msg.get("rtt_ms")
            if isinstance(rtt, (int, float)) and 0 <= rtt < 5000:
                latency_ema = rtt if latency_ema is None else 0.8 * latency_ema + 0.2 * rtt
            store = app[POLICY]
            policy, phash = store.policy, store.hash
            t0 = time.perf_counter()
            try:
                dec = await loop.run_in_executor(app[EXECUTOR], app[BRAIN].decide, msg, policy, latency_ema or 0.0)
            except Exception as e:  # a model failure must not kill the connection
                log.exception("brain failed")
                await ws.send_json({"v": PROTOCOL_VERSION, "type": "error", "error": f"{type(e).__name__}: {e}", "tick": msg["tick"]})
                continue
            out = {"v": PROTOCOL_VERSION, "type": "act", "tick": msg["tick"], "action": dec.action,
                   "hold_ticks": policy["hold_ticks"], "text": dec.text, "answers": dec.answers, "reason": dec.reason,
                   "ttc": dec.ttc, "infer_ms": dec.infer_ms, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                   "policy": phash}
            await ws.send_json(out)
            if decision_log:
                decision_log.write(json.dumps({"t": time.time(), "policy": phash, "obs": msg, **asdict(dec)}) + "\n")

        elif msg.get("type") == "episode_end":
            msg.pop("v", None)
            msg["policy"] = app[POLICY].hash
            msg["brain"] = app[BRAIN].name
            app[EPISODES].append(msg)
            log.info("episode seed=%s score=%s outcome=%s crashed_on=%s", msg.get("seed"), msg.get("score"),
                     msg.get("outcome"), msg.get("crashed_on"))
    return ws


def create_app(brain: Brain, store: PolicyStore, *, max_age_ticks: int = 12, decision_log=None,
               allow_any_host: bool = False) -> web.Application:
    app = web.Application(client_max_size=MAX_MSG_BYTES)
    app[BRAIN] = brain
    app[POLICY] = store
    app[EPISODES] = deque(maxlen=1000)
    app[EXECUTOR] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="brain")  # one model, one queue
    app[SETTINGS] = {"max_age_ticks": max_age_ticks, "decision_log": decision_log, "allow_any_host": allow_any_host}
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/policy", get_policy)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web/", WEB_DIR, show_index=False)

    async def shutdown(app: web.Application) -> None:
        app[EXECUTOR].shutdown(wait=False, cancel_futures=True)
    app.on_cleanup.append(shutdown)
    return app


def add_brain_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--brain", choices=("laya", "oracle"), default="laya", help="who answers the questions (default: laya)")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY), help="policy JSON file")
    ap.add_argument("--device", default=None, help="force a torch device for Laya: cuda, mps or cpu")
    ap.add_argument("--oracle-delay-ms", type=float, default=0.0, help="simulated model latency for the oracle brain")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_brain_args(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-age-ticks", type=int, default=12, help="clients drop answers older than this")
    ap.add_argument("--log-decisions", default=None, help="append every decision as JSONL to this file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    policy = load_policy(args.policy)
    log.info("loading brain %s ...", args.brain)
    brain = make_brain(args.brain, device=args.device, oracle_delay_ms=args.oracle_delay_ms, checkpoint=policy["checkpoint"])
    decision_log = open(args.log_decisions, "a", buffering=1) if args.log_decisions else None
    app = create_app(brain, PolicyStore(policy), max_age_ticks=args.max_age_ticks, decision_log=decision_log,
                     allow_any_host=args.host not in ("127.0.0.1", "localhost", "::1"))
    print(f"Dino decision server ({brain.name}) on http://{args.host}:{args.port}  (ws: /ws, health: /health)", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
