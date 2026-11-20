"""The Jouvea test on current OpenRouter models: one sheet, OCR then parsing with the same model.

    uv run python tools/sweep_openrouter_jouvea.py --plan      list the models and the worst-case cost
    uv run python tools/sweep_openrouter_jouvea.py --run       run (resumable; finished models are skipped)
    uv run python tools/sweep_openrouter_jouvea.py --run --nr  the same models that can reason, with reasoning
                                                               off (or the most minimal setting OpenRouter accepts)
    uv run python tools/sweep_openrouter_jouvea.py --report    rebuild summary.json and index.html only

Each model goes through the app's own routes (POST /process stage=ocr, then
POST /v1/process/finish), so the label collage is what the model reads, exactly
as in the app. A private `vvl serve` is started on its own port with a scratch
home; it uses the real credentials file for the OpenRouter key (never printed).

Settings for this test (diagnostic switches of vvl/engines/openrouter.py):
reasoning on at each model's own default effort, reasoning text returned, image
detail high, and every request and full OpenRouter answer appended to
openrouter_exchanges.jsonl in the output folder.

Models: every model that reads images, is not free and not a :batch variant, and is
  * the newest numbered model of each current line of Anthropic (Fable, Opus, Sonnet, Haiku)
    and OpenAI (Astra, Sol, Terra, Luna, mini), or
  * a Google Gemini 4 model, Gemini 3.8 Flash, or Gemini 3.1 Pro (the VVGO baselines), or
  * from any other owner and released in the last 365 days.

PASS = formatted_json.scientificName contains "Jouvea" (the VoucherVisionGO rule).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"
DEFAULT_OUT = BENCH / "sweep_openrouter_jouvea_test_2026-11-19"
IMAGE = BENCH / "extra_images" / "MICH_16205594_Poaceae_Jouvea_pilosa_full.jpg"
PROMPT = "SLTPvM_full.yaml"
BIG_LABS = ("anthropic", "openai", "google")
EXTRA_GOOGLE = ("google/gemini-3.8-flash", "google/gemini-3.1-pro-preview")
LOG_NAME = "openrouter_exchanges.jsonl"
# The no-reasoning runs ([NR] in the page) keep their own results folder and log.
NR_LOG_NAME = "openrouter_exchanges_nr.jsonl"
RUNS = {False: ("models", LOG_NAME), True: ("models_nr", NR_LOG_NAME)}
MAX_OUT_TOKENS = 16384            # the engine's limit for models that think
WORKERS = 4
# The three original VoucherVisionGO runs of this sheet (bench/vvgo_thinking_budget_jouvea.json).
BASELINES = (("gemini-3.1-pro", "Low"), ("gemini-3.8-flash", "Low"), ("gemini-3.1-flash-lite", "High"))
OWNER_COLORS = {
    "anthropic": "#D97757", "openai": "#10A37F", "google": "#4796E3", "mistralai": "#FA520F", "qwen": "#615CED",
    "meta": "#0668E1", "meta-llama": "#0668E1", "x-ai": "#8a8f98", "deepseek": "#4D6BFE", "z-ai": "#2f6fed",
    "moonshotai": "#16a3b7", "nvidia": "#76B900", "amazon": "#FF9900", "cohere": "#39594D", "microsoft": "#00A4EF",
    "xiaomi": "#FF6900", "tencent": "#0052D9", "perplexity": "#20808D",
}


def safe_name(slug: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", slug).strip("_")


# Current lines of the two labs whose catalogs are full of older versions: the newest
# numbered model of each line is run (not the "-pro" variants, not the "latest" pointers).
CURRENT_LINES = {
    "anthropic": [r"^anthropic/claude-fable-[\d.]+$", r"^anthropic/claude-opus-[\d.]+$",
                  r"^anthropic/claude-sonnet-[\d.]+$", r"^anthropic/claude-haiku-[\d.]+$"],
    "openai": [r"^openai/gpt-[\d.]+-astra$", r"^openai/gpt-[\d.]+-sol$", r"^openai/gpt-[\d.]+-terra$",
               r"^openai/gpt-[\d.]+-luna$", r"^openai/gpt-[\d.]+-mini$"],
}


def newest_of_lines(catalog: dict) -> set[str]:
    newest = set()
    for patterns in CURRENT_LINES.values():
        for pattern in patterns:
            matches = [m for m in catalog["models"] if re.match(pattern, m["id"]) and m["vision"]]
            if matches:
                newest.add(max(matches, key=lambda m: m["created"] or 0)["id"])
    return newest


def choose_models(catalog: dict) -> list[dict]:
    now = catalog.get("fetched_at") or time.time()
    current = newest_of_lines(catalog)
    chosen = []
    for m in catalog["models"]:
        slug = m["id"]
        if not m["vision"] or m["free"] or ":batch" in slug:
            continue
        alias = slug.startswith("~")
        if m["owner"] in BIG_LABS:
            keep = slug in current or slug.startswith("google/gemini-4") or slug in EXTRA_GOOGLE
            why = ("newest of its line" if slug in current
                   else ("Gemini 4 family" if "gemini-4" in slug else "VVGO baseline model"))
        else:
            keep = not alias and bool(m["created"]) and now - m["created"] < 365 * 86400
            why = "released in the last year"
        if keep:
            chosen.append(dict(m, why=why))
    chosen.sort(key=lambda m: (m["owner"] not in BIG_LABS, m["owner_label"].lower(), -(m["created"] or 0)))
    return chosen


def worst_case_usd(m: dict) -> float:
    """Both steps hitting the output limit, with generous input sizes."""
    limit = min(MAX_OUT_TOKENS, m.get("max_output_tokens") or MAX_OUT_TOKENS)
    return ((6000 + 4000) * m["price_in"] + 2 * limit * m["price_out"]) / 1_000_000


# ------------------------------------------------------------------- server --

def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 30)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


class Server:
    def __init__(self, home: Path, port: int, log_path: Path, reasoning: str = "default"):
        self.home, self.log_path, self.reasoning = home, log_path, reasoning
        self.port = free_port(port)
        self.token = secrets.token_urlsafe(32)
        self.base = f"http://127.0.0.1:{self.port}"
        self.popen: subprocess.Popen | None = None
        self.client = httpx.Client(base_url=self.base, headers={"X-VVL-Token": self.token},
                                   timeout=httpx.Timeout(1500.0, connect=10.0))

    def start(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        settings_path = self.home / "settings.yaml"
        settings_path.write_text("# OpenRouter Jouvea sweep (tools/sweep_openrouter_jouvea.py)\n"
                                 "enrichment:\n  elevation: false\n  wfo_source: 'off'\n"
                                 f"openrouter:\n  concurrency: {WORKERS}\n")
        (self.home / "prompts_user").mkdir(exist_ok=True)
        env = dict(os.environ)
        env.pop("VVL_PRIVATE_PATH", None)        # the real credentials file holds the OpenRouter key
        env.update({
            "VVL_HOME": str(self.home), "VVL_SETTINGS_PATH": str(settings_path),
            "VVL_PROMPTS_USER_DIR": str(self.home / "prompts_user"), "VVL_TOKEN": self.token,
            "VVL_OWNER_PID": str(os.getpid()),
            "VVL_OPENROUTER_REASONING": self.reasoning, "VVL_OPENROUTER_INCLUDE_REASONING": "1",
            "VVL_OPENROUTER_IMAGE_DETAIL": "high", "VVL_OPENROUTER_LOG": str(self.log_path),
        })
        log = open(self.home / "server_stdout.log", "a")
        self.popen = subprocess.Popen(["uv", "run", "--project", str(REPO), "vvl", "serve", "--port", str(self.port)],
                                      cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.popen.poll() is not None:
                raise RuntimeError(f"vvl serve exited; see {self.home / 'server_stdout.log'}")
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError("vvl serve did not answer /health within 180 s")

    def stop(self) -> None:
        if not self.popen or self.popen.poll() is not None:
            return
        try:
            os.killpg(self.popen.pid, signal.SIGTERM)
            self.popen.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass

    def call(self, method: str, path: str, **kw) -> dict:
        response = self.client.request(method, path, **kw)
        try:
            body = response.json()
        except ValueError:
            body = {"message": response.text[:500]}
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code} {body.get('error_code', '')}: {body.get('message', body)}")
        return body


# ---------------------------------------------------------------------- run --

def strip_images(value, saved: dict, out_dir: Path):
    """Replace large base64 images in a response by a note; the first one is saved once as collage.jpg."""
    if isinstance(value, dict):
        return {k: strip_images(v, saved, out_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_images(v, saved, out_dir) for v in value]
    if isinstance(value, str) and len(value) > 20000 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value[:2000]):
        import base64

        with saved["lock"]:
            if not saved.get("done"):
                try:
                    (out_dir / "collage.jpg").write_bytes(base64.b64decode(value))
                    saved["done"] = True
                except ValueError:
                    pass
        return f"<image, {len(value) * 3 // 4:,} bytes; the label collage is saved once as collage.jpg>"
    return value


def run_model(server: Server, m: dict, out_dir: Path, saved: dict) -> dict:
    model_id = m["model_id"]
    out = {"id": m["id"], "model_id": model_id, "name": m["name"], "owner": m["owner"], "owner_label": m["owner_label"],
           "why": m["why"], "price_in": m["price_in"], "price_out": m["price_out"], "created": m["created"],
           "reasoning_supported": m["reasoning"], "private": m.get("private"),
           "started": dt.datetime.now().isoformat(timespec="seconds")}
    common = {"engines": model_id, "llm_model": model_id, "prompt": PROMPT, "include_wfo": "false", "include_cop90": "false"}
    t0 = time.monotonic()
    try:
        with open(IMAGE, "rb") as handle:
            staged = server.call("POST", "/process", data=dict(common, stage="ocr"),
                                 files={"file": (IMAGE.name, handle, "image/jpeg")})
        t1 = time.monotonic()
        out["ocr_seconds"] = round(t1 - t0, 2)
        out["ocr_text"] = staged.get("ocr", "")
        out["ocr_info"] = next(iter((staged.get("ocr_info") or {}).values()), {}) or {}
        out["ocr_info"].pop("ocr_text", None)
        if "stage_token" not in staged:
            raise RuntimeError("the OCR step gave no stage token")
        result = server.call("POST", "/v1/process/finish", json={
            "stage_token": staged["stage_token"], "llm_model": model_id, "prompt": PROMPT,
            "include_wfo": False, "include_cop90": False})
        out["parse_seconds"] = round(time.monotonic() - t1, 2)
        out["vvl_output"] = strip_images(result, saved, out_dir)
        out["parsing_info"] = result.get("parsing_info") or {}
        out["record"] = result.get("formatted_json")
        out["text_collage"] = (result.get("success") or {}).get("text_collage")
        out["vvl_cost_usd"] = result.get("total_request_cost_usd")
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{exc}"[:1200]
    out["total_seconds"] = round(time.monotonic() - t0, 2)
    name = str((out.get("record") or {}).get("scientificName") or "") if isinstance(out.get("record"), dict) else ""
    out["scientificName"] = name
    out["jouvea_pass"] = "jouvea" in name.lower()
    out["ocr_read_name"] = "jouvea pilosa" in " ".join(str(out.get("ocr_text") or "").lower().split())
    return out


def spent_usd(log_path: Path) -> float:
    total = 0.0
    if log_path.exists():
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    usage = (json.loads(line).get("response") or {}).get("usage") or {}
                    total += float(usage.get("cost") or 0.0)
                except (ValueError, AttributeError, TypeError):
                    continue
    return total


# ------------------------------------------------------------------- report --

def read_log(log_path: Path) -> dict:
    """{model slug: {step: [exchange summaries]}} from the exchange log."""
    by_model: dict = {}
    if not log_path.exists():
        return by_model
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            response = entry.get("response") if isinstance(entry.get("response"), dict) else {}
            request = entry.get("request") or {}
            choice = (response.get("choices") or [{}])[0] if isinstance(response.get("choices"), list) else {}
            message = choice.get("message") or {}
            usage = response.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning = message.get("reasoning") or ""
            if not reasoning and isinstance(message.get("reasoning_details"), list):
                reasoning = "\n".join(str(d.get("text") or d.get("summary") or "") for d in message["reasoning_details"]
                                      if isinstance(d, dict)).strip()
            sent = {k: v for k, v in request.items() if k not in ("messages", "model", "stream")}
            if "response_format" in sent:
                sent["response_format"] = "json_schema (the prompt's fields)"
            error = response.get("error") if isinstance(response.get("error"), dict) else None
            by_model.setdefault(entry.get("model"), {}).setdefault(entry.get("step") or "?", []).append({
                "time": entry.get("time"), "http_status": entry.get("http_status"), "sent": sent,
                "provider": response.get("provider"), "served_model": response.get("model"),
                "finish_reason": choice.get("finish_reason"), "native_finish_reason": choice.get("native_finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"), "cost": usage.get("cost"),
                "reasoning_text": str(reasoning)[:6000], "reasoning_chars": len(str(reasoning)),
                "error": error.get("message") if error else (
                    str(entry.get("response"))[:500] if entry.get("http_status") != 200 else None),
            })
    return by_model


def baseline_rows() -> list[dict]:
    path = BENCH / "vvgo_thinking_budget_jouvea.json"
    try:
        rows = json.loads(path.read_text())["rows"]
    except (OSError, ValueError, KeyError):
        return []
    num = lambda v: float(str(v).replace(",", "").replace("$", "") or 0)  # noqa: E731
    out = []
    for model, level in BASELINES:
        row = next((r for r in rows if r.get("Model") == model and r.get("Level") == level), None)
        if not row:
            continue
        out.append({
            "baseline": True, "id": f"vvgo/{model}", "name": f"{model} ({level.lower()} thinking)", "owner": "google",
            "owner_label": "Google", "why": "original VoucherVisionGO run", "ok": row.get("Result") == "OK",
            "jouvea_pass": row.get("Jouvea Test") == "PASS", "scientificName": "", "ocr_read_name": None,
            "cost": num(row.get("Overall Cost (USD)")), "total_seconds": None,
            "tokens_in": int(num(row.get("OCR In")) + num(row.get("LLM In"))),
            "tokens_out": int(num(row.get("OCR Out")) + num(row.get("LLM Out"))),
            "reasoning_tokens": int(num(row.get("Think Total"))), "providers": ["Google (VoucherVisionGO)"],
            "steps": {}, "notes": [f"Thinking level {level}; run through VoucherVisionGO, not OpenRouter"],
        })
    return out


def build_report(out_dir: Path, models: list[dict], running: bool = False) -> dict:
    rows = []
    logs = {nr: read_log(out_dir / RUNS[nr][1]) for nr in RUNS}
    for nr, m in [(nr, m) for nr in (False, True) for m in models]:
        path = out_dir / RUNS[nr][0] / f"{safe_name(m['id'])}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        steps = logs[nr].get(m["id"], {})
        calls = [c for step in steps.values() for c in step]
        good = [c for c in calls if c["http_status"] == 200 and c.get("completion_tokens") is not None]
        notes = []
        if any(c.get("finish_reason") == "length" for c in good):
            notes.append("hit the output token limit")
        if len(steps.get("parse", [])) > 1:
            notes.append("parsing was sent twice (the JSON schema was refused the first time)")
        if r.get("ok") and not isinstance(r.get("record"), dict):
            notes.append("no record returned")
        asked = None
        if nr:
            # What OpenRouter accepted for "reasoning off": the setting of the last answered request.
            efforts = sorted({(c["sent"].get("reasoning") or {}).get("effort") for c in good} - {None},
                             key=lambda e: ("none", "minimal", "low").index(e) if e in ("none", "minimal", "low") else 9)
            asked = efforts[-1] if efforts else None
            used = sum(int(c.get("reasoning_tokens") or 0) for c in good)
            if asked and asked != "none":
                notes.append(f"reasoning could not be turned off; ran at \"{asked}\"")
            elif asked == "none" and used:
                notes.append("asked for no reasoning, but the model still reasoned")
        rows.append({
            "baseline": False, "nr": nr, "reasoning_setting": asked,
            "id": r["id"] + (" [NR]" if nr else ""), "name": r["name"] + (" [NR]" if nr else ""),
            "owner": r["owner"], "owner_label": r["owner_label"],
            "why": r.get("why"), "ok": bool(r.get("ok")), "error": r.get("error"),
            "jouvea_pass": bool(r.get("jouvea_pass")), "scientificName": r.get("scientificName") or "",
            "ocr_read_name": r.get("ocr_read_name"), "ocr_text": (r.get("ocr_text") or "")[:3000],
            "record": r.get("record") if isinstance(r.get("record"), dict) else None,
            "cost": round(sum(float(c.get("cost") or 0) for c in calls), 6), "vvl_cost": r.get("vvl_cost_usd"),
            "total_seconds": r.get("total_seconds"), "ocr_seconds": r.get("ocr_seconds"), "parse_seconds": r.get("parse_seconds"),
            "tokens_in": sum(int(c.get("prompt_tokens") or 0) for c in good),
            "tokens_out": sum(int(c.get("completion_tokens") or 0) for c in good),
            "reasoning_tokens": sum(int(c.get("reasoning_tokens") or 0) for c in good),
            "reasoning_supported": r.get("reasoning_supported"), "private": r.get("private"),
            "providers": sorted({c["provider"] for c in good if c.get("provider")}),
            "price_in": r.get("price_in"), "price_out": r.get("price_out"), "created": r.get("created"),
            "steps": steps, "notes": notes,
        })
    data = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"), "image": IMAGE.name, "prompt": PROMPT,
        "models_planned": len(models), "rows": baseline_rows() + rows, "owner_colors": OWNER_COLORS,
        "running": running,
        "spent_usd": round(spent_usd(out_dir / LOG_NAME), 4),
        "spent_nr_usd": round(spent_usd(out_dir / NR_LOG_NAME), 4),
        "settings": "Reasoning on at each model's default effort (reasoning text returned), image detail high, "
                    "label collage as the image, OCR then parsing with the same model, prompt " + PROMPT + ".",
    }
    (out_dir / "summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    template = (REPO / "tools" / "sweep_openrouter_jouvea_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (out_dir / "index.html").write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    return data


# --------------------------------------------------------------------- main --

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--home", default=str(Path.home() / ".local" / "share" / "VoucherVisionLocal-orsweep"))
    parser.add_argument("--port", type=int, default=8674)
    parser.add_argument("--only", default="", help="comma list of OpenRouter model names")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N models")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--nr", action="store_true",
                        help="run the models that can reason with reasoning off ([NR] rows in the page)")
    parser.add_argument("--running", action="store_true",
                        help="with --report: mark the page as still running (it then reloads itself every minute)")
    parser.add_argument("--max-usd", type=float, default=25.0, help="stop starting new models once this much is spent")
    args = parser.parse_args()

    out_dir = Path(args.out)
    sub_dir, log_name = RUNS[args.nr]
    (out_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    if args.report and plan_path.exists():
        models = json.loads(plan_path.read_text())["models"]
        data = build_report(out_dir, models, running=args.running)
        print(f"Report: {out_dir / 'index.html'} ({len(data['rows'])} rows)")
        return 0

    server = Server(Path(args.home), args.port, out_dir / log_name, reasoning="off" if args.nr else "default")
    print(f"Starting the sweep server on port {server.port}", flush=True)
    server.start()
    try:
        catalog = server.call("GET", "/v1/openrouter/catalog")
        if not catalog["configured"]:
            raise SystemExit("No OpenRouter key is saved.")
        models = choose_models(catalog)
        plan_path.write_text(json.dumps({"made": dt.datetime.now().isoformat(timespec="seconds"), "models": models}, indent=1))
        todo = models
        if args.nr:
            # Only models that can reason, and that answered in the default run.
            def answered(m):
                path = out_dir / "models" / f"{safe_name(m['id'])}.json"
                return path.exists() and json.loads(path.read_text()).get("ok")
            todo = [m for m in models if m["reasoning"] and answered(m)]
        if args.only:
            wanted = {w.strip() for w in args.only.split(",") if w.strip()}
            todo = [m for m in models if m["id"] in wanted]
        if args.limit:
            todo = todo[:args.limit]
        worst = sum(worst_case_usd(m) for m in todo)
        print(f"{len(todo)} of {len(models)} models{' (no reasoning)' if args.nr else ''}; worst case if every call hits the output limit: ${worst:.2f}", flush=True)
        if args.plan:
            for m in todo:
                print(f"  {m['owner_label']:<16} {m['id']:<52} ${m['price_in']:>6.2f} / ${m['price_out']:>6.2f}  {m['why']}")
            return 0
        if args.report:
            build_report(out_dir, models)
            return 0

        def pending(m):
            path = out_dir / sub_dir / f"{safe_name(m['id'])}.json"
            if not path.exists():
                return True
            return args.retry_errors and not json.loads(path.read_text()).get("ok")

        todo = [m for m in todo if pending(m)]
        saved = {"lock": threading.Lock(), "done": (out_dir / "collage.jpg").exists()}
        done = 0
        lock = threading.Lock()

        def work(m):
            nonlocal done
            if spent_usd(out_dir / log_name) >= args.max_usd:
                return None
            result = run_model(server, m, out_dir, saved)
            (out_dir / sub_dir / f"{safe_name(m['id'])}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
            with lock:
                done += 1
                state = "PASS" if result["jouvea_pass"] else ("FAIL" if result["ok"] else "ERROR")
                print(f"[{done:>3}/{len(todo)}] {state:<5} {m['id']:<50} {result['total_seconds']:>6.1f} s  "
                      f"{(result.get('scientificName') or result.get('error') or '')[:70]}", flush=True)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(work, todo))
        data = build_report(out_dir, models)
        ran = [r for r in data["rows"] if not r["baseline"] and bool(r.get("nr")) == args.nr]
        print(f"Done: {sum(r['jouvea_pass'] for r in ran)} of {len(ran)} pass; spent "
              f"${data['spent_nr_usd' if args.nr else 'spent_usd']:.4f}. "
              f"Report: {out_dir / 'index.html'}", flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
