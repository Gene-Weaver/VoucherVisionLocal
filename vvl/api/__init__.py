"""The /v1 API (docs/API.md sections 6, 9 and 10), as Flask blueprints."""

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from vvl.api import (app_info, detector, engine, generation, jobs, keys, metrics, models, openrouter, outputs,
                         prompts_routes, settings_routes, system, url_names, viewer)

    for module in (app_info, system, keys, models, settings_routes, prompts_routes, engine, jobs, metrics,
                   url_names, generation, openrouter, detector, outputs, viewer):
        app.register_blueprint(module.bp)
