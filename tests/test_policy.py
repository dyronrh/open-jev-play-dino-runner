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


def obs(kind="cactus_small_1", x=200.0, speed=10.0, jumping=False, ducking=False, falling=False):
    return {"tick": 1, "speed": speed,
            "dino": {"x": 50, "width": 44, "elev": 0, "jumping": jumping, "falling": falling, "ducking": ducking},
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
    # gap = 200 + 0.5 * 17 - 94 = 114.5 px at 10 px/tick -> 11.45 ticks -> "close" with the default bins
    s = render(obs(x=200), policy)
    assert s.truth == {"obstacle": "ground", "distance": "close"}
    assert policy["phrases"]["obstacle"]["cactus_small_1"] in s.text
    assert policy["phrases"]["distance"]["close"] in s.text
    assert s.text.startswith(policy["phrases"]["dino"]["ground"])


def test_render_compensates_latency(policy):
    # 100 ms of latency is 6 ticks: 11.45 ticks away now becomes 5.45 by the time the answer lands
    assert render(obs(x=200), policy, latency_ms=100).truth["distance"] == "touching"


def test_render_aims_into_wide_obstacles(policy):
    narrow, wide = obs(x=200), obs(x=200)
    wide["obstacles"][0]["width"] = 75  # a group of three tall cacti
    front = {**policy, "width_frac": 0.0}
    assert render(narrow, front).ttc == render(wide, front).ttc == 10.6
    assert render(narrow, policy).ttc == 11.45
    assert render(wide, policy).ttc == 14.35  # the wide group is described later, so jumped later


def test_predict_landing_describes_the_dino_as_it_will_be(policy):
    from dino_agent.policy import ticks_to_land
    assert 34 <= ticks_to_land(0.001, -10.3) <= 38  # a full jump lasts about 35 ticks
    landing = obs(jumping=True, falling=True)
    landing["dino"].update(elev=18, vy=5)  # lands in 4 ticks
    on = {**policy, "predict_landing": True}
    assert render(landing, on, latency_ms=100).dino == "ground"   # 6 ticks of latency: landed by then
    assert render(landing, on, latency_ms=33).dino == "air"       # 2 ticks: still in the air
    assert render(landing, {**policy, "predict_landing": False}, latency_ms=100).dino == "air"
    no_vy = obs(jumping=True)
    assert render(no_vy, on, latency_ms=100).dino == "air"        # older pages send no vy


def test_width_frac_is_optional(policy):
    p = copy.deepcopy(policy)
    del p["width_frac"]
    validate_policy(p)
    assert render(obs(x=200), p).ttc == 10.6  # front edge, as before the key existed


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


FALLING = "The dino is falling back down at the end of a jump."
FAST_FALL = {"action": "duck", "when": {"obstacle": ["ground"], "distance": ["close"]}, "dino": ["falling"]}


def test_fast_fall_only_on_the_way_down(policy):
    policy = copy.deepcopy(policy)
    policy["phrases"]["dino"]["falling"] = FALLING
    policy["rules"].append(FAST_FALL)
    validate_policy(policy)
    rising = render(obs(jumping=True), policy)
    falling = render(obs(jumping=True, falling=True), policy)
    assert (rising.dino, falling.dino) == ("air", "falling")
    assert policy["phrases"]["dino"]["falling"] in falling.text
    assert decide(ans("ground", "close"), rising, policy)[0] == "none"   # never drop onto what it is clearing
    assert decide(ans("ground", "close"), falling, policy)[0] == "duck"  # fast fall to be ready to jump again
    assert decide(ans("ground", "touching"), falling, policy)[0] == "none"  # may be the obstacle underneath


def test_falling_state_is_optional(policy):
    assert "falling" not in policy["phrases"]["dino"]  # measured worse in the default policy
    p = copy.deepcopy(policy)
    assert render(obs(jumping=True, falling=True), p).dino == "air"
    p["rules"].append(FAST_FALL)
    with pytest.raises(PolicyError, match="phrases.dino.falling"):
        validate_policy(p)


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
    ({"width_frac": 1.5}, "width_frac"),
    ({"predict_landing": "yes"}, "predict_landing"),
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
    assert report["scenes"] == len(policy["phrases"]["dino"]) * (1 + len(OBSTACLE_KINDS) * 4)
    assert report["accuracy"] == {"obstacle": 1.0, "distance": 1.0}
