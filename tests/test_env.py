from evoplan.envs import ToyKitchen, make_stream, parse_schedule
from evoplan.envs.toy_kitchen import make_task


def env():
    return ToyKitchen(seed=0, fault_rate=0.0, relocate_rate=0.0)


def test_heat_task_solvable_in_domain_a():
    e = env()
    e.reset(make_task("t", "heat", "apple", "table", "A"))
    for action in ["go fridge", "open fridge", "take apple", "go microwave", "open microwave",
                   "use microwave", "go table"]:
        _, ok, done, _ = e.step(action)
        assert ok and not done, action
    _, ok, done, _ = e.step("put apple table")
    assert ok and done


def test_broken_microwave_in_domain_b():
    e = env()
    e.reset(make_task("t", "heat", "potato", "table", "B"))
    for action in ["go cabinet", "open cabinet", "take potato", "go microwave", "open microwave"]:
        assert e.step(action)[1], action
    obs, ok, _, info = e.step("use microwave")
    assert not ok and info["event"] == "broken"
    assert e.hint_valid({"task": "heat", "tool": "microwave"}) is False
    assert e.hint_valid({"task": "heat", "tool": "stove"}) is True


def test_closed_container_hides_contents():
    e = env()
    e.reset(make_task("t", "pick", "apple", "table", "A"))
    _, _, _, info = e.step("go fridge")
    assert info["closed"] and info["visible"] is None
    _, ok, _, _ = e.step("take apple")
    assert not ok


def test_drift_uses_hidden_layout():
    phases = parse_schedule("A:2,A@B:2")
    assert [p.label for p in phases] == ["A", "A@B"]
    stream = make_stream(phases, seed=1)
    drifted = [t for p, t in stream if p == 1]
    assert all(t.domain == "A" and t.layout == "B" for t in drifted)
    e = env()
    e.reset(make_task("t", "pick", "apple", "table", "A", layout="B"))
    assert e.hint_valid({"obj": "apple", "loc": "cabinet"}) is True   # B's home, despite label A


def test_streams_are_reproducible():
    phases = parse_schedule("A:5,B:5")
    assert make_stream(phases, 3) == make_stream(phases, 3)
    assert make_stream(phases, 3) != make_stream(phases, 4)
