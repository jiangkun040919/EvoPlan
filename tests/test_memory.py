from evoplan.memory import Experience, ExperienceStore, TrustConfig, trust_score


def make(kind="location", key=None, ctx="domain=A obj=apple", t=0):
    key = key or {"obj": "apple", "loc": "fridge"}
    return Experience(id=f"{kind}:{sorted(key.items())}", text="x", kind=kind, key=key, context=ctx,
                      created=t, last_validated=t)


def test_contradiction_is_local_to_context():
    store = ExperienceStore()
    exp = store.add(make(), ctx="A")
    for i in range(5):
        store.record_outcome([exp.id], [], success=True, now=i, ctx="A")
    before_a, before_b = store.trust(exp, 5, "A"), store.trust(exp, 5, "B")
    store.record_outcome([exp.id], [exp.id], success=False, now=6, ctx="B")
    after_a, after_b = store.trust(exp, 6, "A"), store.trust(exp, 6, "B")
    assert after_b < 0.55 < after_a          # gated in B, still usable in A
    assert before_b - after_b > before_a - after_a


def test_global_trust_ablation_spreads_contradictions():
    store = ExperienceStore(TrustConfig(contextual=False))
    exp = store.add(make(), ctx="A")
    store.record_outcome([exp.id], [exp.id], success=False, now=1, ctx="B")
    assert store.trust(exp, 1, "A") == store.trust(exp, 1, "B")


def test_unvalidated_trust_decays_toward_half():
    exp = make()
    exp.successes = 10
    fresh, stale = trust_score(exp, 0), trust_score(exp, 500)
    assert fresh > stale > 0.5


def test_merge_counts_support_and_local_confirmation():
    store = ExperienceStore()
    store.add(make(t=0), ctx="A")
    merged = store.add(make(ctx="domain=B obj=apple", t=3), ctx="B")
    assert len(store) == 1 and merged.support == 2
    assert merged.by_context == {"A": [1, 0, 0], "B": [1, 0, 0]}
    assert "domain=B" in merged.context and merged.last_validated == 3


def test_retrieve_gate_and_one_hint_per_slot():
    store = ExperienceStore()
    good = store.add(make(key={"obj": "apple", "loc": "fridge"}), ctx="A")
    bad = store.add(make(key={"obj": "apple", "loc": "sink"}), ctx="A")
    store.record_outcome([bad.id], [bad.id], success=False, now=1, ctx="A")
    gated = store.retrieve("domain=A kind=heat obj=apple", now=1, ctx="A")
    assert [r.exp.id for r in gated] == [good.id]
    naive = store.retrieve("domain=A kind=heat obj=apple", now=1, use_trust=False, gate=False, ctx="A")
    assert len(naive) == 1  # only one location per object, even without gating


def test_save_load_roundtrip(tmp_path):
    store = ExperienceStore()
    store.add(make(), ctx="A")
    store.save(tmp_path / "m.jsonl")
    loaded = ExperienceStore.load(tmp_path / "m.jsonl")
    assert [e.to_dict() for e in loaded] == [e.to_dict() for e in store]
