"""Decide which engines to run: which model on which graphics card, with how many request slots.

`plan_layout()` is a pure function. Given the models in use, the memory each
card can spare, and a few rules, it returns a `Layout`: a list of engines to
run. The engine pool (pool.py) makes the running engines match it.

How it decides
--------------
1. **Everything at once, if it fits ("shared").** Each model gets one engine on
   the card with the most room. If every model can be placed, the models stay
   loaded together and no request ever waits for a model swap.
2. **Then use what is left.** Spare memory goes, one step at a time, to the
   model with the least capacity for its workload: either another request slot
   on an engine it already has, or another engine on a card it is not on yet.
   That is how one model ends up on every card, and how a small model gets
   several slots or copies.
3. **One model at a time, if they do not fit together ("exclusive").** Only the
   model that is needed now is loaded, on every card it fits on. Batches then
   run all OCR first and all parsing second, so models are swapped once per
   batch instead of twice per image.
4. A model that is too large for any single card gets one engine spread over
   all cards.

A request slot costs memory for its context window; an engine copy costs the
model's weights again. The rules (`Rules`) say how many of each are worth
having; their defaults come from measurements with `tools/bench_layouts.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024 ** 3


@dataclass(frozen=True)
class ModelNeed:
    """One model the plan must serve."""
    model_id: str
    base_bytes: int            # weights + vision projector + fixed runtime overhead
    slot_bytes: int            # memory for one request slot (context window)
    roles: tuple[str, ...] = ("ocr", "parse")
    label: str = ""

    def unit_bytes(self, slots: int) -> int:
        return self.base_bytes + max(1, slots) * self.slot_bytes


@dataclass(frozen=True)
class Card:
    index: int
    budget_bytes: int          # memory VoucherVisionLocal may use on this card


@dataclass(frozen=True)
class Rules:
    # Request slots per engine. `fixed_slots` is the user's explicit setting;
    # otherwise the planner grows from `min_slots` up to `max_slots`.
    min_slots: int = 1
    max_slots: int = 4
    fixed_slots: int | None = None
    # Engine copies of one model on the same card (1 = never duplicate weights
    # on a card; slots are used instead).
    max_copies_per_card: int = 1
    # Upper bound on request slots across all engines, so a tiny model on a
    # large card does not fan out past what the rest of the pipeline can feed.
    max_total_slots: int = 16
    # Relative workload per role, used to split spare memory between an OCR
    # model and a parsing model (parsing writes more tokens per image).
    role_weight: dict = field(default_factory=lambda: {"ocr": 1.0, "parse": 2.0})


@dataclass
class UnitSpec:
    """One engine to run."""
    model_id: str
    gpu_devices: list[int] | None    # card indexes; [] = processor only; None = engine default
    parallel: int
    bytes: int = 0

    def key(self) -> tuple:
        devices = None if self.gpu_devices is None else tuple(self.gpu_devices)
        return (self.model_id, devices, self.parallel)


@dataclass
class Layout:
    mode: str                         # none | single | shared | exclusive
    units: list[UnitSpec]
    active: str | None = None         # in exclusive mode, the model that is loaded
    fits: bool = True                 # False when even the minimum does not fit the budgets
    waiting: list[str] = field(default_factory=list)   # models not loaded in this layout

    def models(self) -> list[str]:
        seen: list[str] = []
        for unit in self.units:
            if unit.model_id not in seen:
                seen.append(unit.model_id)
        return seen

    def slots_for(self, model_id: str) -> int:
        return sum(u.parallel for u in self.units if u.model_id == model_id)

    def bytes_by_card(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for unit in self.units:
            devices = unit.gpu_devices or []
            for index in devices:
                out[index] = out.get(index, 0) + unit.bytes // max(1, len(devices))
        return out


def _weight(model: ModelNeed, rules: Rules) -> float:
    return sum(rules.role_weight.get(role, 1.0) for role in model.roles) or 1.0


def _start_slots(rules: Rules) -> int:
    return rules.fixed_slots or rules.min_slots


def _place_one_each(models: list[ModelNeed], cards: list[Card], rules: Rules):
    """One engine per model, largest first, each on the card with the most room.

    Returns (units, remaining budgets) or None when some model fits on no card.
    """
    remaining = {c.index: c.budget_bytes for c in cards}
    units: list[UnitSpec] = []
    slots = _start_slots(rules)
    for model in sorted(models, key=lambda m: m.unit_bytes(slots), reverse=True):
        need = model.unit_bytes(slots)
        best = max(remaining, key=lambda i: (remaining[i], -i), default=None)
        if best is None or need > remaining[best]:
            return None
        remaining[best] -= need
        units.append(UnitSpec(model.model_id, [best], slots, need))
    return units, remaining


def _grow(units: list[UnitSpec], remaining: dict[int, int], models: dict[str, ModelNeed], rules: Rules) -> None:
    """Spend spare memory on more slots and more engines, evening out capacity."""
    def total_slots() -> int:
        return sum(u.parallel for u in units)

    def capacity(model_id: str) -> float:
        slots = sum(u.parallel for u in units if u.model_id == model_id)
        return slots / _weight(models[model_id], rules)

    while total_slots() < rules.max_total_slots:
        options = []
        for model_id, model in models.items():
            # (a) a new engine on a card that has room and fewer copies than allowed
            for index, room in remaining.items():
                copies = sum(1 for u in units if u.model_id == model_id and u.gpu_devices == [index])
                need = model.unit_bytes(_start_slots(rules))
                if copies < rules.max_copies_per_card and need <= room:
                    # Prefer cards the model is not on yet, then the roomiest card.
                    options.append((capacity(model_id), 0 if copies == 0 else 2, -room, "unit", model_id, index, need))
            # (b) one more slot on an existing single-card engine
            if rules.fixed_slots is None:
                for unit in units:
                    if unit.model_id != model_id or unit.gpu_devices is None or len(unit.gpu_devices) != 1:
                        continue
                    index = unit.gpu_devices[0]
                    if unit.parallel < rules.max_slots and model.slot_bytes <= remaining.get(index, 0):
                        options.append((capacity(model_id), 1, unit.parallel, "slot", model_id, id(unit), model.slot_bytes))
        if not options:
            return
        options.sort(key=lambda o: o[:3])
        _cap, _kind_rank, _tie, kind, model_id, target, need = options[0]
        if kind == "unit":
            slots = _start_slots(rules)
            units.append(UnitSpec(model_id, [target], slots, models[model_id].unit_bytes(slots)))
            remaining[target] -= need
        else:
            unit = next(u for u in units if id(u) == target)
            unit.parallel += 1
            unit.bytes += need
            remaining[unit.gpu_devices[0]] -= need


def _exclusive(model: ModelNeed, cards: list[Card], rules: Rules) -> tuple[list[UnitSpec], bool]:
    """Every card that can hold the model runs it; else one engine over all cards."""
    slots = _start_slots(rules)
    need = model.unit_bytes(slots)
    remaining = {c.index: c.budget_bytes for c in cards}
    fitting = [i for i, room in remaining.items() if need <= room]
    if fitting:
        best = max(fitting, key=lambda i: (remaining[i], -i))
        units = [UnitSpec(model.model_id, [best], slots, need)]
        remaining[best] -= need
        _grow(units, remaining, {model.model_id: model}, rules)
        return units, True
    # Too large for one card: spread over all of them (llama.cpp splits the layers).
    indexes = sorted(remaining)
    fits = need <= sum(remaining.values())
    return [UnitSpec(model.model_id, indexes, slots, need)], fits


def plan_layout(models: list[ModelNeed], cards: list[Card], *, rules: Rules | None = None,
                active: str | None = None, backend: str = "cuda") -> Layout:
    """The engines to run for `models` on `cards`.

    `active` is the model a request needs right now; it matters only when the
    models cannot be loaded together. `backend` "cpu" and "metal" have no
    separate cards: they get one engine per model while memory allows (`cards`
    then holds a single pseudo-card with the memory budget, or is empty when
    the budget is unknown).
    """
    rules = rules or Rules()
    seen: dict[str, ModelNeed] = {}
    for model in models:
        seen.setdefault(model.model_id, model)
    unique = list(seen.values())
    if not unique:
        return Layout("none", [])
    if active is None or active not in seen:
        active = unique[0].model_id

    if backend in ("cpu", "metal") or not cards:
        # No per-card placement: the engine uses the processor or unified memory.
        devices = [] if backend == "cpu" else None
        slots = _start_slots(rules) if rules.fixed_slots else min(rules.max_slots, max(rules.min_slots, 2))
        budget = cards[0].budget_bytes if cards else None
        together = sum(m.unit_bytes(slots) for m in unique)
        if len(unique) == 1 or (budget is not None and together <= budget):
            units = [UnitSpec(m.model_id, devices, slots, m.unit_bytes(slots)) for m in unique]
            fits = budget is None or together <= budget
            return Layout("single" if len(unique) == 1 else "shared", units, active=active, fits=fits)
        model = seen[active]
        fits = budget is None or model.unit_bytes(slots) <= budget
        return Layout("exclusive", [UnitSpec(model.model_id, devices, slots, model.unit_bytes(slots))], active=active,
                      fits=fits, waiting=[m.model_id for m in unique if m.model_id != active])

    placed = _place_one_each(unique, cards, rules)
    if placed is not None:
        units, remaining = placed
        _grow(units, remaining, seen, rules)
        order = {m.model_id: i for i, m in enumerate(unique)}
        units.sort(key=lambda u: (order[u.model_id], u.gpu_devices or []))
        return Layout("single" if len(unique) == 1 else "shared", units, active=active)

    units, fits = _exclusive(seen[active], cards, rules)
    units.sort(key=lambda u: u.gpu_devices or [])
    mode = "single" if len(unique) == 1 else "exclusive"
    return Layout(mode, units, active=active, fits=fits,
                  waiting=[m.model_id for m in unique if m.model_id != active])
