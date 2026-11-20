"""`local:<build_id>` models: the managed llama.cpp engine behind the OpenAI-compatible engine.

The resolver calls `availability()` (is the build downloaded and the engine
installed?) and `create()`. Each OCR/parse call holds a supervisor lease, which
starts the engine or switches it to this model first (serialized), then sends
one request through `OpenAICompatEngine`.
"""

from __future__ import annotations

from vvl.engines.base import EngineCapabilities, EngineError, ModelRef, OCRRequest, ParseRequest, ResolveContext

CAPABILITIES = EngineCapabilities(vision=True, structured_output=True, thinking_levels=())

REASON_DOWNLOADING = "This model is still downloading."
REASON_QUEUED = "This model is waiting to download."
REASON_PARTIAL = "The download of this model is incomplete; resume it on the Models tab."
REASON_ERROR = "The last download of this model failed; try again on the Models tab."
REASON_NOT_INSTALLED = "Download this model first (Models tab)."
REASON_NO_ENGINE = "Install the llama.cpp engine first (Engine tab)."
REASON_NEITHER = "Download this model and install the llama.cpp engine first (Models and Engine tabs)."


def reason_for(install_state: str, engine_installed: bool) -> str | None:
    """unavailable_reason for a build, or None when it can run."""
    if install_state == "installed":
        return None if engine_installed else REASON_NO_ENGINE
    if install_state == "downloading":
        return REASON_DOWNLOADING
    if install_state == "queued":
        return REASON_QUEUED
    if install_state == "partial":
        return REASON_PARTIAL
    if install_state == "error":
        return REASON_ERROR
    return REASON_NOT_INSTALLED if engine_installed else REASON_NEITHER


def _runtime(context: ResolveContext):
    return getattr(context, "runtime", None)


def _states(ref: ModelRef, context: ResolveContext):
    """(family, build, install_state, engine_installed)."""
    found = context.local_models.find_build(ref.name)
    family, build = found
    runtime = _runtime(context)
    if runtime is not None:
        state = runtime.models.status(build)["install_state"]
        engine_installed = bool(runtime.installer.installed_ids())
        return family, build, state, engine_installed
    # No runtime (e.g. `vvl doctor`): read the disk directly.
    from vvl import settings as settings_mod
    from vvl.jobs import JobManager
    from vvl.llama.installer import EngineInstaller
    from vvl.llama.lock import load_lock
    from vvl.local_models import disk_status

    current = context.settings or settings_mod.load()
    models_dir = current.get("models_dir") or settings_mod.load()["models_dir"]
    state = disk_status(models_dir, build)["state"]
    installer = EngineInstaller(JobManager(), lock=load_lock, models_dir=lambda: models_dir, report=lambda: None)
    return family, build, state, bool(installer.installed_ids())


def availability(ref: ModelRef, context: ResolveContext) -> tuple[bool, str | None]:
    _family, _build, state, engine_installed = _states(ref, context)
    reason = reason_for(state, engine_installed)
    return reason is None, reason


class LocalEngine:
    capabilities = CAPABILITIES
    notice = None

    def __init__(self, ref: ModelRef, family, build, runtime):
        self.model_id = ref.model_id
        self.label = build.label
        self.family = family
        self.build = build
        self.runtime = runtime

    def _call(self, step: str, request):
        from vvl.engines.openai_compat import OpenAICompatConfig, OpenAICompatEngine, StepOptions
        from vvl.llama.supervisor import EngineNotReady

        from vvl import generation
        from vvl.llama import inference as inference_mod

        # Per request, from the user's choices on the VVL tab and the model's
        # entry in model_settings.json: output token limit, the model card's
        # sampling defaults, and the reasoning switch (chat_template_kwargs or a
        # prompt suffix). Engine-level settings (context, image token bounds,
        # server_args) apply at the next engine start.
        infer = inference_mod.settings_for(self.family, self.build)
        try:
            current = self.runtime.settings()
        except Exception:  # noqa: BLE001
            current = {}
        sent = generation.step_request(self.model_id, step, current, self.runtime.local_registry())
        options = StepOptions(max_tokens=sent["max_tokens"], sampling=dict(sent["sampling"]),
                              image_first=infer["ocr"].get("image_first", True) is not False,
                              extra_body=dict(sent["extra_body"]), prompt_suffix=sent["prompt_suffix"])
        try:
            with self.runtime.supervisor.lease(self.model_id) as handle:
                config = OpenAICompatConfig(
                    base_url=handle.base_url,
                    api_key=handle.api_key,
                    model=handle.model_name,
                    model_id=self.model_id,
                    label=self.label,
                    capabilities=CAPABILITIES,
                    ocr=options,
                    parse=options,
                    timeout_seconds=float(infer.get("request_timeout_seconds") or 900),
                    max_retries=1,
                    service_label="The local engine",
                )
                engine = OpenAICompatEngine(config)
                return engine.ocr(request) if step == "ocr" else engine.parse(request)
        except EngineNotReady as exc:
            raise EngineError(f"The local engine is not ready: {exc.message}", code="ENGINE_NOT_READY",
                              status=503) from None

    def ocr(self, request: OCRRequest):
        return self._call("ocr", request)

    def parse(self, request: ParseRequest):
        return self._call("parse", request)


def create(ref: ModelRef, context: ResolveContext) -> LocalEngine:
    from vvl.engines.base import ModelNotAvailable

    runtime = _runtime(context)
    if runtime is None:
        raise ModelNotAvailable(ref.model_id, "The local engine is only available inside the running app.")
    family, build = context.local_models.find_build(ref.name)
    return LocalEngine(ref, family, build, runtime)
