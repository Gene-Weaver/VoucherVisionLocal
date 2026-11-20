"""Where the selected local models will go: graphics memory per card.

`memory_plan()` answers "how much graphics memory will be set aside for the
models chosen on the VVL tab, and on which cards". The resource monitor
(GET /v1/metrics) draws its "planned" segment from it.

It asks the engine pool for the layout it aims for (`EnginePool.plan`), so the
monitor always shows what the pool will really do:

- `single`     one local model, on every card it fits on;
- `shared`     the OCR model and the parsing model loaded side by side;
- `exclusive`  the models do not fit together, so one is loaded at a time and
               each card needs room for the larger one, not for both.

Memory an engine already holds is counted in a card's `engine_bytes`, so only
what is still to be loaded shows up as `planned_bytes`.
"""

from __future__ import annotations

from vvl.llama.pool import selected_local_models

MODE_NOTES = {
    "none": "No local model is selected, so no graphics memory is planned.",
    "single": "One local model is used. It runs on every graphics card it fits on.",
    "shared": "The selected local models fit together, so they stay loaded side by side.",
    "exclusive": "The selected local models do not fit in graphics memory together, so one is loaded at a time. "
                 "Each card needs room for the larger one, not for both.",
}
CPU_NOTE = " The installed engine build runs on the processor and uses no graphics memory."


def _unit_cards(unit, gpus: list[dict]) -> list[int]:
    if unit.gpu_devices is None:
        # The engine decides (Apple Metal): unified-memory cards are the ones it uses.
        return [g["index"] for g in gpus if g.get("kind") == "unified" or g.get("vendor") == "apple"]
    return [i for i in unit.gpu_devices if any(g["index"] == i for g in gpus)]


def _bytes_by_card(layout, gpus: list[dict], model_id: str | None = None) -> dict[int, int]:
    out: dict[int, int] = {}
    for unit in layout.units if layout else []:
        if model_id is not None and unit.model_id != model_id:
            continue
        cards = _unit_cards(unit, gpus)
        for index in cards:
            out[index] = out.get(index, 0) + unit.bytes // max(1, len(cards))
    return out


def memory_plan(pool, selection: dict, registry, report: dict | None, gpus: list[dict]) -> dict:
    """The memory plan for the selected local models.

    `gpus` is the live card list of GET /v1/metrics (`index`, `vram_free_bytes`,
    `engine_bytes`, ...). Returns::

        {"mode": "none" | "single" | "shared" | "exclusive", "note": str,
         "models": [{"model_id", "label", "roles", "required_bytes", "gpu_indices", "loaded"}],
         "gpus": {index: {"planned_bytes", "planned_for": [{"model_id", "label", "bytes"}],
                          "planned_fits"}}}
    """
    selected = selected_local_models(selection)
    status = pool.status()
    loaded = set(status.get("loaded_models") or [])
    engines = status.get("engines") or []
    # Free memory per card as it would be with none of our engines running.
    free = {g["index"]: g["vram_free_bytes"] + (g.get("engine_bytes") or 0)
            for g in gpus if isinstance(g.get("vram_free_bytes"), int)}
    layout = pool.plan(free=free) if selected else None
    on_cpu = (status.get("backend") == "cpu")

    # In exclusive mode each model gets the cards to itself in turn: look at the
    # layout every model would have while it is the loaded one.
    layouts = {}
    for model_id, _roles in selected:
        if layout is None:
            layouts[model_id] = None
        elif layout.mode == "exclusive" and model_id not in layout.models():
            layouts[model_id] = pool.plan(active=model_id, free=free)
        else:
            layouts[model_id] = layout

    models = []
    shares: dict[str, dict[int, int]] = {}
    tight: set[int] = set()
    for model_id, roles in selected:
        found = registry.find_build(model_id.split(":", 1)[1])
        own = layouts.get(model_id)
        by_card = _bytes_by_card(own, gpus, model_id)
        running = [e for e in engines if e.get("model_id") == model_id]
        if running:
            indices = sorted({i for e in running for i in (e.get("gpu_devices") or [])})
        else:
            indices = sorted(by_card)
        required = sum(u.bytes for u in own.units if u.model_id == model_id) if own else 0
        models.append({"model_id": model_id, "label": found[1].label if found else model_id, "roles": list(roles),
                       "required_bytes": required or None, "gpu_indices": indices, "loaded": model_id in loaded})
        if model_id not in loaded and by_card:
            shares[model_id] = by_card
            if own is not None and not own.fits:
                tight.update(by_card)      # even the smallest layout exceeds the memory budget

    mode = layout.mode if layout is not None else "none"
    if not models:
        mode = "none"
    per_card = {}
    for g in gpus:
        index = g["index"]
        waiting = [{"model_id": m["model_id"], "label": m["label"], "bytes": shares[m["model_id"]][index]}
                   for m in models if m["model_id"] in shares and index in shares[m["model_id"]]]
        if mode == "exclusive":
            # One model at a time: the card must hold the largest of them. What the
            # engine holds now is released first, so only the amount beyond it is new.
            wanted = max((w["bytes"] for w in waiting), default=0)
            planned = max(0, wanted - (g.get("engine_bytes") or 0))
        else:
            planned = sum(w["bytes"] for w in waiting)
        free_now = g.get("vram_free_bytes")
        per_card[index] = {
            "planned_bytes": planned,
            "planned_for": waiting,
            "planned_fits": True if planned == 0 else (None if free_now is None
                                                       else (planned <= free_now and index not in tight)),
        }
    note = MODE_NOTES.get(mode, "") + (CPU_NOTE if on_cpu and models else "")
    return {"mode": mode, "note": note, "models": models, "gpus": per_card}
