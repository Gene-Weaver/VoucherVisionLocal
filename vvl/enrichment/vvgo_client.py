"""Data-enrichment helpers served by the hosted VoucherVisionGO API.

Elevation: the public `GET {vvgo}/elevation?lat=&lon=` route (COP90 DEM). No
key is sent; only the two coordinates leave this computer. Short timeout and
graceful failure: any problem returns None and the record keeps "".

World Flora Online: not available in Phase 1 (VoucherVisionGO's `/wfo` route
and the optional local database arrive in Phase 3). `wfo_lookup()` reports
that instead of calling anything.
"""

from __future__ import annotations

import logging
import math

import requests

from vvl.registry import loader

logger = logging.getLogger(__name__)

# (connect, read). A cold VoucherVisionGO instance took ~4.7 s to answer in testing.
ELEVATION_TIMEOUT_SECONDS = (3.05, 8.0)
WFO_UNAVAILABLE_REASON = "World Flora Online lookups are not available in this version yet."
_FALLBACK_BASE_URL = "https://vouchervision-go-738307415303.us-central1.run.app"


def vvgo_base_url(providers_path=None) -> str:
    try:
        provider = loader.find_provider("vvgo", providers_path)
        if provider is not None:
            return provider.base_url
    except loader.RegistryError:
        pass
    return _FALLBACK_BASE_URL


def _coordinate(value, low, high):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or not (low <= number <= high):
        return None
    return number


def elevation_m(lat, lon, *, base_url: str | None = None, timeout=ELEVATION_TIMEOUT_SECONDS,
                session: requests.Session | None = None):
    """COP90 elevation in meters for (lat, lon), or None on any problem."""
    lat_v = _coordinate(lat, -90.0, 90.0)
    lon_v = _coordinate(lon, -180.0, 180.0)
    if lat_v is None or lon_v is None:
        return None
    url = f"{(base_url or vvgo_base_url()).rstrip('/')}/elevation"
    http = session or requests
    try:
        response = http.get(
            url,
            params={"lat": f"{lat_v:.6f}", "lon": f"{lon_v:.6f}"},
            timeout=timeout,
            allow_redirects=False,
            headers={"Accept": "application/json"},
        )
    except requests.RequestException as exc:
        logger.info("Elevation lookup failed (%s).", type(exc).__name__)
        return None
    if response.status_code != 200:
        logger.info("Elevation lookup answered HTTP %s.", response.status_code)
        return None
    try:
        value = response.json().get("elevation_m")
    except (ValueError, AttributeError):
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


def elevation_for_record(record, **kwargs):
    """Look up elevation from a parsed record's decimal coordinates (VVGO field names)."""
    if not isinstance(record, dict):
        return None
    lat = record.get("decimalLatitude") or record.get("decimal_latitude")
    lon = record.get("decimalLongitude") or record.get("decimal_longitude")
    if lat in (None, "") or lon in (None, ""):
        return None
    return elevation_m(lat, lon, **kwargs)


def wfo_lookup(record, source: str = "off") -> dict:
    """Phase 1 stub: World Flora Online is not available yet."""
    return {"available": False, "source": source, "reason": WFO_UNAVAILABLE_REASON, "result": None}
