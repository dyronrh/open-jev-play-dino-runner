from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dino_agent.arena import Arena
from dino_agent.brains import OracleBrain
from dino_agent.policy import load_policy
from dino_agent.server import PolicyStore, create_app

from .conftest import needs_chromium

ROOT = Path(__file__).resolve().parent.parent
POLICY = load_policy(ROOT / "policies" / "default.json")
OBS = {"v": 1, "type": "obs", "tick": 5, "speed": 10.0, "rtt_ms": None,
       "dino": {"x": 50, "width": 44, "elev": 0, "jumping": False, "ducking": False},
       "obstacles": [{"id": 1, "kind": "cactus_small_1", "x": 200, "width": 17, "height": 35, "elev": 0}]}


@pytest.fixture
async def client():
    app = create_app(OracleBrain(), PolicyStore(POLICY))
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    body = await r.json()
    assert body["ok"] and body["brain"]["name"] == "oracle" and body["protocol"] == 1


async def test_ws_hello_obs_act(client):
    async with client.ws_connect("/ws") as ws:
        hello = await ws.receive_json()
        assert hello["type"] == "hello" and hello["max_age_ticks"] == 12
        await ws.send_json(OBS)
        act = await ws.receive_json()
        assert act["type"] == "act" and act["tick"] == 5
        assert act["action"] == "jump"  # a small cactus 10.6 ticks away is "close"
        assert act["answers"]["obstacle"]["choice"] == "ground"
        assert act["policy"] == hello["policy"]


async def test_ws_rejects_bad_messages(client):
    async with client.ws_connect("/ws") as ws:
        await ws.receive_json()
        await ws.send_str("not json")
        assert (await ws.receive_json())["error"] == "invalid JSON"
        await ws.send_json({**OBS, "v": 99})
        assert "protocol version" in (await ws.receive_json())["error"]
        await ws.send_json({**OBS, "speed": "fast"})
        assert "speed" in (await ws.receive_json())["error"]
        await ws.send_json(OBS)  # the connection survives errors
        assert (await ws.receive_json())["type"] == "act"


async def test_ws_rejects_foreign_origin(client):
    r = await client.get("/ws", headers={"Origin": "https://evil.example", "Upgrade": "websocket",
                                          "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
                                          "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
    assert r.status == 403


async def test_episode_end_is_recorded(client):
    async with client.ws_connect("/ws") as ws:
        await ws.receive_json()
        await ws.send_json({"v": 1, "type": "episode_end", "seed": 3, "score": 42, "outcome": "crash"})
        await ws.send_json(OBS)
        await ws.receive_json()
    body = await (await client.get("/health")).json()
    assert body["episodes"] == 1


async def test_index_serves_the_original_game(client):
    html = await (await client.get("/")).text()
    assert '/web/original/index.js' in html and 'id="audio-resources"' in html
    js = await client.get("/web/original/index.js")
    assert js.status == 200 and "The Chromium Authors" in await js.text()
    assert (await client.get("/web/original/assets/default_100_percent/100-offline-sprite.png")).status == 200


@needs_chromium
async def test_original_game_end_to_end():
    """The real page in headless Chromium: the original game, driven over the WebSocket."""
    async with Arena(OracleBrain(), POLICY) as arena:
        report = await arena.evaluate(POLICY, [1, 2], max_seconds=10, parallel=2)
    assert report["errors"] == []
    assert report["episodes"] == 2
    assert all(s["score"] > 0 for s in report["per_seed"])
    assert all(s["outcome"] in ("crash", "timeout") for s in report["per_seed"])
