from evoplan.agent import METHODS, Agent
from evoplan.envs import ToyKitchen, make_stream, parse_schedule
from evoplan.envs.toy_kitchen import make_task
from evoplan.llm import ScriptedLLM
from evoplan.memory.reflection import LLMReflector, RuleReflector
from evoplan.memory.store import ExperienceStore
from evoplan.planning.planner import RulePlanner
from evoplan.policy import HeuristicPolicy, LLMPolicy, parse_action
from evoplan.types import Episode, Step


def agent(name):
    cfg = METHODS[name]
    store = ExperienceStore() if cfg.use_memory else None
    return Agent(cfg, HeuristicPolicy(), RulePlanner(), RuleReflector(), store)


def run_stream(name, schedule="A:15,A@B:15", seed=0):
    a = agent(name)
    env = ToyKitchen(seed=seed, fault_rate=0.0, relocate_rate=0.0)
    eps = [a.run(env, t, i, seed) for i, (_, t) in enumerate(make_stream(parse_schedule(schedule), seed))]
    return a, eps


def test_no_exp_never_uses_memory():
    a, eps = run_stream("no_exp")
    assert a.store is None and all(not e.used_experiences for e in eps)


def test_experience_saves_steps_in_a_stable_domain():
    _, base = run_stream("no_exp", "A:30")
    _, evo = run_stream("evoplan", "A:30")
    assert sum(len(e.steps) for e in evo[10:]) < sum(len(e.steps) for e in base[10:])


def test_drift_produces_contradictions_and_replans():
    a, eps = run_stream("evoplan")
    drifted = eps[15:]
    assert any(e.contradicted for e in drifted)
    assert sum(e.replans for e in drifted) > 0
    # after the drift, the fact that holds in the new layout outranks the stale one for context A
    for obj, new_home in [("egg", "microwave"), ("lettuce", "fridge")]:
        top = a.store.retrieve(f"domain=A kind=pick obj={obj}", now=30, ctx="A")
        loc_hint = next(r.exp.key["loc"] for r in top if r.exp.kind == "location")
        assert loc_hint == new_home, obj


def test_rule_reflector_extracts_location_and_tool():
    task = make_task("t", "heat", "apple", "table", "A")
    ep = Episode(task=task, method="x", seed=0, index=4, success=True, steps=[
        Step("go fridge", "", True), Step("open fridge", "", True), Step("take apple", "", True),
        Step("go stove", "", True), Step("use stove", "", True)])
    exps = RuleReflector().reflect(ep)
    assert {e.kind: e.key for e in exps} == {"location": {"obj": "apple", "loc": "fridge"},
                                            "tool": {"task": "heat", "tool": "stove"}}


def test_parse_action_and_llm_policy():
    acts = ["go fridge", "open microwave", "take apple"]
    assert parse_action("Thought: ...\nACTION: open microwave", acts) == "open microwave"
    assert parse_action("ACTION: `take apple`.", acts) == "take apple"
    env = ToyKitchen(seed=0, fault_rate=0.0, relocate_rate=0.0)
    task = make_task("t", "pick", "apple", "table", "A")
    llm = ScriptedLLM(["ACTION: go fridge", "ACTION: open fridge", "ACTION: take apple",
                       "ACTION: go table", "ACTION: put apple table", "INSIGHT: the apple is in the fridge"])
    a = Agent(METHODS["evoplan_no_replan"], LLMPolicy(llm, env.admissible_actions), RulePlanner(),
              LLMReflector(llm), ExperienceStore())
    ep = a.run(env, task, 0, 0)
    assert ep.success and len(ep.steps) == 5
    assert ep.prompt_tokens > 0
    assert [e.text for e in a.store] == ["the apple is in the fridge"]
