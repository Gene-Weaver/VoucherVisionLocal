"""The engine layout planner (vvl/llama/planner.py): pure decisions, no engines."""

from __future__ import annotations

from vvl.llama.planner import GIB, Card, ModelNeed, Rules, plan_layout


def model(name, weights_gb, slot_gb=1.0, roles=("ocr", "parse")):
    return ModelNeed(f"local:{name}", int(weights_gb * GIB), int(slot_gb * GIB), roles=roles, label=name)


def cards(*gb):
    return [Card(i, int(v * GIB)) for i, v in enumerate(gb)]


def summary(layout):
    return sorted((u.model_id.split(":")[1], tuple(u.gpu_devices or []), u.parallel) for u in layout.units)


def test_no_models():
    layout = plan_layout([], cards(40, 40))
    assert layout.mode == "none" and layout.units == []


def test_one_model_runs_on_every_card_with_several_slots():
    layout = plan_layout([model("small", 8)], cards(46, 46))
    assert layout.mode == "single" and layout.fits
    assert summary(layout) == [("small", (0,), 4), ("small", (1,), 4)]
    # Never two copies of the same model on one card: slots are used instead.
    assert len({tuple(u.gpu_devices) for u in layout.units}) == len(layout.units)


def test_large_model_gets_one_engine_per_card_it_fits_on():
    layout = plan_layout([model("big", 36)], cards(46, 44))
    assert summary(layout) == [("big", (0,), 4), ("big", (1,), 4)]
    layout = plan_layout([model("big", 36)], cards(46, 30))
    assert summary(layout) == [("big", (0,), 4)], "the second card has no room for another copy"


def test_slots_stop_where_memory_runs_out():
    layout = plan_layout([model("big", 36, slot_gb=2)], cards(41))
    assert summary(layout) == [("big", (0,), 2)]      # 36 + 2*2 = 40 <= 41, a third slot would not fit


def test_fixed_slots_are_respected():
    layout = plan_layout([model("small", 8)], cards(46, 46), rules=Rules(fixed_slots=2))
    assert summary(layout) == [("small", (0,), 2), ("small", (1,), 2)]


def test_total_slot_limit():
    layout = plan_layout([model("tiny", 2, slot_gb=0.5)], cards(46, 46, 46, 46, 46, 46), rules=Rules(max_total_slots=8))
    assert sum(u.parallel for u in layout.units) == 8


def test_two_models_that_fit_are_loaded_together():
    ocr, parse = model("ocr", 4, roles=("ocr",)), model("parse", 10, roles=("parse",))
    layout = plan_layout([ocr, parse], cards(46, 46))
    assert layout.mode == "shared" and layout.fits and layout.waiting == []
    assert set(layout.models()) == {"local:ocr", "local:parse"}
    # Both models end up on both cards, and parsing (the heavier step) is not short of slots.
    assert {tuple(u.gpu_devices) for u in layout.units if u.model_id == "local:ocr"} == {(0,), (1,)}
    assert {tuple(u.gpu_devices) for u in layout.units if u.model_id == "local:parse"} == {(0,), (1,)}
    assert layout.slots_for("local:parse") >= layout.slots_for("local:ocr")
    for index, used in layout.bytes_by_card().items():
        assert used <= 46 * GIB


def test_two_models_on_one_small_card_take_turns():
    # The 16 GB card from the question: 9 GB + 15 GB do not fit together.
    ocr, parse = model("ocr", 8, roles=("ocr",)), model("parse", 14, roles=("parse",))
    layout = plan_layout([ocr, parse], cards(16), active="local:parse")
    assert layout.mode == "exclusive" and layout.fits
    assert layout.models() == ["local:parse"] and layout.waiting == ["local:ocr"]
    assert summary(layout) == [("parse", (0,), 2)]
    other = plan_layout([ocr, parse], cards(16), active="local:ocr")
    assert other.models() == ["local:ocr"] and other.waiting == ["local:parse"]
    assert sum(u.parallel for u in other.units) == 4


def test_two_models_two_cards_one_each_when_only_that_fits():
    a, b = model("a", 30, roles=("ocr",)), model("b", 30, roles=("parse",))
    layout = plan_layout([a, b], cards(40, 40))
    assert layout.mode == "shared"
    assert sorted((u.model_id, tuple(u.gpu_devices)) for u in layout.units) == \
        [("local:a", (1,)), ("local:b", (0,))] or sorted((u.model_id, tuple(u.gpu_devices)) for u in layout.units) == \
        [("local:a", (0,)), ("local:b", (1,))]


def test_model_too_large_for_any_card_spreads_over_all():
    layout = plan_layout([model("huge", 60)], cards(40, 40))
    assert layout.mode == "single" and layout.fits
    assert summary(layout) == [("huge", (0, 1), 1)]
    layout = plan_layout([model("huge", 90)], cards(40, 40))
    assert not layout.fits and summary(layout) == [("huge", (0, 1), 1)]


def test_cpu_and_metal_have_no_card_placement():
    layout = plan_layout([model("m", 8)], [], backend="cpu")
    assert layout.mode == "single" and layout.units[0].gpu_devices == []
    layout = plan_layout([model("m", 8)], cards(24), backend="metal")
    assert layout.units[0].gpu_devices is None and layout.fits
    a, b = model("a", 8), model("b", 14)
    together = plan_layout([a, b], cards(64), backend="metal")
    assert together.mode == "shared" and len(together.units) == 2
    turns = plan_layout([a, b], cards(16), backend="metal", active="local:b")
    assert turns.mode == "exclusive" and turns.models() == ["local:b"] and turns.waiting == ["local:a"]
    unknown = plan_layout([a, b], [], backend="metal", active="local:a")
    assert unknown.mode == "exclusive", "unknown memory: never load two models at once"


def test_same_model_listed_twice_counts_once():
    m = model("m", 8)
    layout = plan_layout([m, m], cards(46))
    assert layout.mode == "single" and len(layout.units) == 1


def test_plan_is_stable_for_small_memory_changes():
    a = plan_layout([model("m", 8)], cards(46.0, 46.0))
    b = plan_layout([model("m", 8)], cards(45.7, 46.2))
    assert summary(a) == summary(b)
