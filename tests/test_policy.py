import copy
import json
from pathlib import Path

import pytest

from dino_agent.policy import (OBSTACLE_KINDS, PolicyError, apply_patch, decide, distance_bin, load_policy,
                               policy_hash, render, validate_policy)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def policy():
    return load_policy(ROOT / "policies" / "default.json")


def obs(kind="cactus_small_1", x=200.0, speed=10.0, jumping=False, ducking=False):
    return {"tick": 1, "speed": speed,
            "dino": {"x": 50, "width": 44, "elev": 0, "jumping": jumping, "ducking": ducking},
            "obstacles": [{"id": 3, "kind": kind, "x": x, "width": 17, "height": 35, "elev": 0}]}


def test_default_policy_is_valid(policy):
    validate_policy(policy)


def test_bins(policy):
    bins = policy["ttc_bins"]
    assert distance_bin(bins["far"], bins) == "far"
    assert distance_bin(bins["near"], bins) == "near"
    assert distance_bin(bins["close"], bins) == "close"
    assert distance_bin(bins["close"] - 0.1, bins) == "touching"


def test_render_uses_phrases_and_truth(policy):
    # gap = 200 - 94 = 106 px at 10 px/tick -> 10.6 ticks -> "close" with the default bins
    s = render(obs(x=200), policy)
    assert s.truth == {"obstacle": "ground", "distance": "close"}
    assert policy["phrases"]["obstacle"]["cactus_small_1"] in s.text
    assert policy["phrases"]["distance"]["close"] in s.text
    assert s.text.startswith(policy["phrases"]["dino"]["ground"])


def test_render_compensates_latency(policy):
    # 100 ms of latency is 6 ticks: 10.6 ticks away now becomes 4.6 by the time the answer lands
    assert render(obs(x=200), policy, latency_ms=100).truth["distance"] == "touching"


def test_render_clear_path(policy):
    o = obs()
    o["obstacles"] = []
    s = render(o, policy)
    assert s.truth["obstacle"] == "none" and policy["phrases"]["clear"] in s.text


@pytest.mark.parametrize("kind,cls", [("bird_mid", "head"), ("bird_high", "sky"), ("bird_low", "ground"), ("cactus_large_3", "ground")])
def test_obstacle_classes(policy, kind, cls):
    assert render(obs(kind=kind), policy).truth["obstacle"] == cls


def test_every_kind_has_a_distinct_phrase(policy):
    phrases = [policy["phrases"]["obstacle"][k] for k in OBSTACLE_KINDS]
    assert len(set(phrases)) == len(phrases)


def ans(obstacle, distance, p=0.9):
    return {"obstacle": {"choice": obstacle, "p": p}, "distance": {"choice": distance, "p": p}}


def test_decide_rules(policy):
    ground = render(obs(), policy)
    assert decide(ans("ground", "close"), ground, policy)[0] == "jump"
    assert decide(ans("ground", "far"), ground, policy)[0] == "none"
    assert decide(ans("head", "touching"), ground, policy)[0] == "duck"
    assert decide(ans("sky", "close"), ground, policy)[0] == "none"
    air = render(obs(jumping=True), policy)
    assert decide(ans("ground", "close"), air, policy)[0] == "none"  # cannot jump mid-air


def test_decide_respects_min_prob(policy):
    s = render(obs(), policy)
    assert decide(ans("ground", "close", p=0.2), s, policy)[0] == "none"


def test_patch_merges_and_validates(policy):
    new = apply_patch(policy, {"ttc_bins": {"close": 4}, "phrases": {"clear": "Nothing ahead."}})
    assert new["ttc_bins"] == {**policy["ttc_bins"], "close": 4}
    assert new["phrases"]["clear"] == "Nothing ahead."
    assert new["phrases"]["dino"] == policy["phrases"]["dino"]
    assert policy_hash(new) != policy_hash(policy)
    assert policy["ttc_bins"]["close"] == 6  # the original is untouched


@pytest.mark.parametrize("patch,msg", [
    ({"ttc_bins": {"near": 50}}, "far > near > close"),
    ({"questions": {"distance": {"criteria": {"sideways": "x"}}}}, "criteria"),
    ({"rules": [{"action": "fly", "when": {"obstacle": ["ground"]}, "dino": ["ground"]}]}, "action"),
    ({"rules": [{"action": "jump", "when": {"obstacle": ["lava"]}, "dino": ["ground"]}]}, "subset"),
    ({"min_prob": {"obstacle": 2}}, "min_prob"),
    ({"surprise": 1}, "unknown top-level"),
    ({"checkpoint": "gpt"}, "checkpoint"),
    ({"phrases": {"clear": ""}}, "phrases.clear"),
])
def test_patch_rejects_bad_changes(policy, patch, msg):
    with pytest.raises(PolicyError, match=msg):
        apply_patch(policy, patch)


def test_policy_file_roundtrip(tmp_path, policy):
    p = copy.deepcopy(policy)
    f = tmp_path / "p.json"
    f.write_text(json.dumps(p))
    assert load_policy(f) == p


def test_eval_questions_enumerates_every_scene(policy):
    from dino_agent.brains import OracleBrain
    from dino_agent.eval_questions import evaluate
    report = evaluate(OracleBrain(), policy)
    assert report["scenes"] == 3 * (1 + len(OBSTACLE_KINDS) * 4)
    assert report["accuracy"] == {"obstacle": 1.0, "distance": 1.0}
