"""vvl/registry/loader.py: the shipped registries load; broken ones are rejected."""

from __future__ import annotations

import copy

import pytest
import yaml

from vvl.registry import loader

VALID_PROVIDER = {
    "id": "x", "label": "X", "role": "model_provider", "kind": "openai_compat", "purpose": "p",
    "sends_images": True, "optional": True, "base_url": "https://api.example.org/v1",
    "fields": [{"name": "api_key", "label": "Key"}],
    "test": {"method": "GET", "url": "{base_url}/models", "auth": "bearer", "ok_status": [200]},
    "models": {"source": "list_endpoint", "url": "{base_url}/models", "auth": "bearer", "id_path": "data[].id"},
    "enabled": True,
}


def test_shipped_providers_load():
    providers = loader.load_providers()
    ids = [p.id for p in providers.providers]
    assert ids == ["vvgo", "gemini", "anthropic", "openai", "openrouter", "huggingface"]
    gemini = loader.find_provider("gemini")
    assert gemini.role == "model_provider"
    assert gemini.test.auth == "query:key"
    assert gemini.models.id_path == "models[].name"
    assert gemini.models.strip_prefix == "models/"
    assert loader.find_provider("huggingface").field_names() == ["token"]
    assert all(p.base_url.startswith("https://") for p in providers.providers)


def test_shipped_local_models_load():
    local = loader.load_local_models()
    family_ids = [f.id for f in local.families]
    assert family_ids[0] == "qwen3-vl"
    qwen = local.families[0]
    assert qwen.status == "verified"
    assert len(qwen.builds) == 7
    family, build = local.find_build("qwen3-vl-8b-q8")
    assert family.id == "qwen3-vl"
    assert build.default_for_test is True
    assert build.total_size_bytes == 8709519456 + 752289728
    assert [f.role for f in build.files] == ["model", "mmproj"]
    planned = [f for f in local.families if f.status == "planned"]
    assert planned and all(f.builds == [] for f in planned)
    assert local.find_build("nope") is None


def test_api_costs_load():
    costs = loader.load_api_costs()
    assert isinstance(costs, dict) and costs
    assert all("in" in v and "out" in v for v in costs.values() if isinstance(v, dict))


def _write(tmp_path, data, name="providers.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(auth_bad=True) or p["test"].update(auth="cookie"),
    lambda p: p.update(base_url="http://api.example.org"),
    lambda p: p.update(role="wizard"),
    lambda p: p.update(id="Bad Id"),
    lambda p: p.update(fields=[]),
    lambda p: p.update(fields=[{"name": "api_key", "label": "a"}, {"name": "api_key", "label": "b"}]),
    lambda p: p["models"].pop("id_path"),
    lambda p: p.pop("sends_images"),
])
def test_invalid_provider_rejected(tmp_path, mutate):
    provider = copy.deepcopy(VALID_PROVIDER)
    mutate(provider)
    path = _write(tmp_path, {"version": 1, "providers": [provider]})
    with pytest.raises(loader.RegistryError):
        loader.load_providers(path)


def test_valid_provider_and_loopback_http(tmp_path):
    provider = copy.deepcopy(VALID_PROVIDER)
    provider["base_url"] = "http://127.0.0.1:9999/"
    loaded = loader.load_providers(_write(tmp_path, {"version": 1, "providers": [provider]}))
    assert loaded.providers[0].base_url == "http://127.0.0.1:9999"


def test_duplicate_provider_ids(tmp_path):
    path = _write(tmp_path, {"version": 1, "providers": [VALID_PROVIDER, VALID_PROVIDER]})
    with pytest.raises(loader.RegistryError, match="duplicate"):
        loader.load_providers(path)


def test_bad_version_and_yaml(tmp_path):
    with pytest.raises(loader.RegistryError):
        loader.load_providers(_write(tmp_path, {"version": 2, "providers": []}))
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [unclosed", encoding="utf-8")
    with pytest.raises(loader.RegistryError):
        loader.load_providers(bad)
    with pytest.raises(loader.RegistryError):
        loader.load_providers(tmp_path / "missing.yaml")


def _local_models():
    with open(loader.DEFAULT_LOCAL_MODELS_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("mutate", [
    lambda d: d["families"][0]["builds"][0]["files"][0].update(sha256="xyz"),
    lambda d: d["families"][0]["builds"][0]["files"][0].update(path="../evil.gguf"),
    lambda d: d["families"][0]["builds"][0]["files"][0].update(size_bytes=0),
    lambda d: d["families"][0]["builds"][0].update(tier="huge"),
    lambda d: d["families"][0]["builds"][1].update(id=d["families"][0]["builds"][0]["id"]),
    lambda d: d["families"][0]["builds"][0].update(files=[d["families"][0]["builds"][0]["files"][1]]),
    lambda d: d["families"][0].update(builds=[]),
    lambda d: d["families"][1].update(id=d["families"][0]["id"]),
    lambda d: d["families"][0]["builds"][0].update(default_for_test=True),
])
def test_invalid_local_models_rejected(tmp_path, mutate):
    data = _local_models()
    mutate(data)
    with pytest.raises(loader.RegistryError):
        loader.load_local_models(_write(tmp_path, data, "models_local.yaml"))
