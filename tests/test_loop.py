"""The ADK loop with a scripted Critic (no LLM, no API key) and the oracle brain (no model)."""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("google.adk")
if shutil.which("node") is None:
    pytest.skip("needs node >= 22", allow_module_level=True)

from google.adk.agents import BaseAgent  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

from dino_agent.arena import Arena  # noqa: E402
from dino_agent.brains import OracleBrain  # noqa: E402
from dino_agent.loop import (BEST_POLICY, HISTORY, PROPOSAL, Evaluator, _event, build_root,  # noqa: E402
                             run_loop)
from dino_agent.policy import apply_patch, load_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class ScriptedCritic(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    proposals: list

    async def _run_async_impl(self, ctx):
        yield _event(self.name, "proposal", {PROPOSAL: self.proposals.pop(0)})


async def test_loop_rejects_invalid_then_accepts_a_fix(tmp_path):
    good = load_policy(ROOT / "policies" / "default.json")
    # Start broken: jump only when the obstacle is "far" (it lands before reaching it and dies).
    broken = apply_patch(good, {"rules": [{"action": "jump", "when": {"obstacle": ["ground"], "distance": ["far"]}, "dino": ["ground"]}]})
    proposals = [
        "not json at all",
        "```json\n" + json.dumps({"rules": good["rules"]}) + "\n```",  # the fix, wrapped the way LLMs do
    ]
    async with Arena(OracleBrain(), broken) as arena:
        evaluator = Evaluator(arena, seeds=[1, 2], max_seconds=10, parallel=2)
        root = build_root(evaluator=evaluator, initial_policy=broken, critic=ScriptedCritic(name="Critic", proposals=proposals),
                          iterations=2, target_score=None, out_dir=tmp_path)
        state = await run_loop(root)

    history = json.loads(state[HISTORY])
    assert [h["accepted"] for h in history] == [True, False, True]  # baseline, invalid, fix
    assert json.loads(state[BEST_POLICY])["rules"] == good["rules"]
    assert (tmp_path / "best_policy.json").exists()
