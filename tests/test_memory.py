import pytest

from evoplan.memory import Experience, ExperienceStore, TrustConfig


def make(kind="location", key=None, ctx="domain=A obj=apple", t=0):
    key = key or {"obj": "apple", "loc": "fridge"}
    return Experience(id=f"{kind}:{sorted(key.items())}", text="x", kind=kind, key=key, context=ctx,
                      created=t, last_evidence=t)


def good_and_bad(cfg=TrustConfig(), n=10):
    """One experience that kept working in A, one that kept failing in A, both last seen at episode n."""
    store = ExperienceStore(cfg)
    good = store.add(make(key={"obj": "apple", "loc": "fridge"}), ctx="A")
    bad = store.add(make(key={"obj": "pear", "loc": "sink"}), ctx="A")
    for i in range(1, n + 1):
        store.record_outcome([good.id], [], success=True, now=i, ctx="A")
        store.record_outcome([bad.id], [], success=False, now=i, ctx="A")
    return store, good, bad


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


def test_evidence_decay_neither_starves_good_nor_revives_bad():
    store, good, bad = good_and_bad()
    now, later = store.estimate(good, 10, "A"), store.estimate(good, 60, "A")
    assert later.mean > 0.55                          # still retrievable after 50 idle episodes
    assert later.uncertainty > now.uncertainty        # but we are less sure about it
    assert store.trust(bad, 60, "A") < 0.5            # a failing experience stays below 0.5
    assert store.estimate(bad, 60, "A").uncertainty > store.estimate(bad, 10, "A").uncertainty


def test_decay_modes_differ_as_designed():
    later = 160
    ev, g_ev, b_ev = good_and_bad(TrustConfig(decay_mode="evidence"))
    zero, g_zero, _ = good_and_bad(TrustConfig(decay_mode="to_zero"))
    half, _, b_half = good_and_bad(TrustConfig(decay_mode="to_half"))
    assert zero.trust(g_zero, later, "A") < 0.55 < ev.trust(g_ev, later, "A")   # to_zero starves good
    assert ev.trust(b_ev, later, "A") < half.trust(b_half, later, "A")          # to_half revives bad faster


def test_old_evidence_stays_discounted_after_new_evidence():
    store = ExperienceStore()
    exp = store.add(make(), ctx="A")
    for i in range(10):
        store.record_outcome([exp.id], [], success=True, now=i, ctx="A")
    store.record_outcome([exp.id], [exp.id], success=False, now=300, ctx="A")
    undiscounted = ExperienceStore(TrustConfig(gamma=1.0))
    assert store.trust(exp, 300, "A") < 0.5 < undiscounted.trust(exp, 300, "A")


def test_quarantine_does_not_expire():
    store = ExperienceStore()
    exp = store.add(make(), ctx="A")
    store.record_outcome([exp.id], [exp.id], success=False, now=1, ctx="A")
    assert exp.quarantined
    assert store.retrieve("domain=A kind=heat obj=apple", now=1000, ctx="A") == []


def test_invalid_decay_mode():
    with pytest.raises(ValueError):
        TrustConfig(decay_mode="linear")


def test_merge_counts_support_and_local_confirmation():
    store = ExperienceStore()
    store.add(make(t=0), ctx="A")
    merged = store.add(make(ctx="domain=B obj=apple", t=3), ctx="B")
    assert len(store) == 1 and merged.support == 2
    assert merged.by_context == {"A": [1, 0, 0], "B": [1, 0, 0]}
    assert "domain=B" in merged.context and merged.last_evidence == 3
    assert merged.evidence == [[0, "A", "create"], [3, "B", "merge"]]


def test_retrieve_gate_and_one_hint_per_slot():
    store = ExperienceStore()
    good = store.add(make(key={"obj": "apple", "loc": "fridge"}), ctx="A")
    bad = store.add(make(key={"obj": "apple", "loc": "sink"}), ctx="A")
    store.record_outcome([bad.id], [bad.id], success=False, now=1, ctx="A")
    gated = store.retrieve("domain=A kind=heat obj=apple", now=1, ctx="A")
    assert [r.exp.id for r in gated] == [good.id]
    assert 0 < gated[0].uncertainty < 0.5
    naive = store.retrieve("domain=A kind=heat obj=apple", now=1, use_trust=False, gate=False, ctx="A")
    assert len(naive) == 1  # only one location per object, even without gating


def test_save_load_roundtrip(tmp_path):
    store, _, _ = good_and_bad(n=3)
    store.save(tmp_path / "m.jsonl")
    loaded = ExperienceStore.load(tmp_path / "m.jsonl")
    assert [e.to_dict() for e in loaded] == [e.to_dict() for e in store]
    assert [loaded.trust(e, 20, "A") for e in loaded] == [store.trust(e, 20, "A") for e in store]
