"""POST /v1/url-names: file names for image URLs, from the real URL name parser."""

from __future__ import annotations

import pytest

from vvl.api import url_names as url_names_module
from vvl.fetch.url_name_parser import resolve_filename_from_response

pytestmark = pytest.mark.usefixtures("no_network")

MICH_URL = ("https://quod.lib.umich.edu/cgi/i/image/api/image/herb00ic:1122841:MICH-V-1122841/full/res:0/0/"
            "native.jpg")

# (url, name): shapes the parser already handles; the route must agree with it.
SHAPES = [
    (MICH_URL, "MICH-V-1122841.jpg"),
    ("https://medialib.naturalis.nl/file/id/L.3800382/format/large", "L.3800382.jpg"),
    ("https://rmh.uwyo.edu/images/jpeg.php?Image=RM0021232.jpg&Width=2000", "RM0021232.jpg"),
    ("https://img.cyverse.org/resize?width=4000&url=https://data.cyverse.org/dav-anon/iplant/projects/"
     "magnoliagrandiFLORA/images/specimens/MISS0055041/MISS0055041.JPG", "MISS0055041.JPG"),
    ("https://swbiodiversity.org/imglib/h_seinet/seinet/KHD/KHD00041/KHD00041592_lg.jpg", "KHD00041592_lg.jpg"),
    ("http://sweetgum.nybg.org/images3/521/802/01449954.jpg", "01449954.jpg"),
    ("https://mediaphoto.mnhn.fr/media/1550169734602JTIb5TZotc7BA4mv", "1550169734602JTIb5TZotc7BA4mv.jpg"),
]


def post(client, auth, body):
    return client.post("/v1/url-names", headers=auth, json=body)


def test_exact_umich_url(client, auth):
    res = post(client, auth, {"urls": [MICH_URL]})
    assert res.status_code == 200
    assert res.get_json() == {"names": [{"url": MICH_URL, "name": "MICH-V-1122841.jpg", "stem": "MICH-V-1122841"}]}


@pytest.mark.parametrize("url,name", SHAPES)
def test_known_shapes_match_the_parser(client, auth, url, name):
    entry = post(client, auth, {"urls": [url]}).get_json()["names"][0]
    assert entry["name"] == name == resolve_filename_from_response(url)
    assert entry["stem"] == name.rsplit(".", 1)[0]
    assert entry["url"] == url


def test_batch_keeps_order_and_duplicates(client, auth):
    urls = [u for u, _ in SHAPES] + [MICH_URL]
    names = post(client, auth, {"urls": urls}).get_json()["names"]
    assert [n["url"] for n in names] == urls
    assert [n["name"] for n in names] == [n for _, n in SHAPES] + ["MICH-V-1122841.jpg"]


def test_invalid_urls_get_the_parsers_fallback_not_an_error(client, auth):
    urls = ["not a url", "", "https://example.org/", MICH_URL]
    res = post(client, auth, {"urls": urls})
    assert res.status_code == 200
    names = res.get_json()["names"]
    for entry, url in zip(names[:3], urls[:3]):
        assert entry["name"] == resolve_filename_from_response(url)
        assert entry["name"].startswith("image_") and entry["name"].endswith(".jpg")
        assert entry["stem"] == entry["name"][:-4]
    assert names[3]["name"] == "MICH-V-1122841.jpg"


def test_empty_list(client, auth):
    assert post(client, auth, {"urls": []}).get_json() == {"names": []}


def test_no_request_is_sent(client, auth, monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise AssertionError("the route must not contact the URL's server")

    for name in ("head", "get", "request"):
        monkeypatch.setattr(requests, name, boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    assert post(client, auth, {"urls": [u for u, _ in SHAPES]}).status_code == 200


def test_a_parser_failure_on_one_url_does_not_fail_the_batch(client, auth, monkeypatch):
    real = url_names_module.resolve_filename_from_response

    def flaky(url, **kwargs):
        if url == "https://example.org/bad":
            raise ValueError("boom")
        return real(url, **kwargs)

    monkeypatch.setattr(url_names_module, "resolve_filename_from_response", flaky)
    names = post(client, auth, {"urls": ["https://example.org/bad", MICH_URL]}).get_json()["names"]
    assert names[0]["name"].startswith("image_") and names[1]["name"] == "MICH-V-1122841.jpg"


def test_requires_the_token(client):
    res = client.post("/v1/url-names", json={"urls": [MICH_URL]})
    assert res.status_code == 401


@pytest.mark.parametrize("body", [
    {}, {"urls": "https://example.org/a.jpg"}, {"urls": [1]}, {"urls": [None]},
    {"urls": ["https://example.org/a.jpg"], "extra": 1},
    {"urls": ["https://example.org/" + "a" * url_names_module.MAX_URL_LENGTH]},
    {"urls": ["https://example.org/a1.jpg"] * (url_names_module.MAX_URLS + 1)},
])
def test_bad_bodies(client, auth, body):
    res = post(client, auth, body)
    assert res.status_code == 400
    assert res.get_json()["error_code"] == "INVALID_INPUT"


def test_the_cap_itself_is_accepted(client, auth):
    urls = [f"https://example.org/img/AB{i:05d}.jpg" for i in range(url_names_module.MAX_URLS)]
    names = post(client, auth, {"urls": urls}).get_json()["names"]
    assert len(names) == url_names_module.MAX_URLS and names[-1]["name"] == "AB04999.jpg"


def test_not_json(client, auth):
    res = client.post("/v1/url-names", headers=auth, data="urls=x")
    assert res.status_code == 400
