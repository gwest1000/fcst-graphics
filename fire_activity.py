#!/usr/bin/env python3
"""Retrieve current BC wildfire incidents with a satellite-hotspot fallback."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import requests


ACTIVE_FIRE_URL = (
    "https://delivery.maps.gov.bc.ca/arcgis/rest/services/mpcm/"
    "bcgwpub/MapServer/502/query"
)
HOTSPOT_URL = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows"
ACTIVE_FIRE_CACHE = Path("data/fire_activity/bcws_active_fires.json")
HOTSPOT_CACHE = Path("data/fire_activity/cwfis_hotspots_24h.json")
ACTIVE_CACHE_MAX_AGE = dt.timedelta(minutes=45)
HOTSPOT_CACHE_MAX_AGE = dt.timedelta(minutes=45)
STALE_CACHE_MAX_AGE = dt.timedelta(hours=12)
REQUEST_EXTENT = (-142.0, -108.0, 45.0, 61.0)
HOTSPOT_GRID_DEGREES = (0.12, 0.09)


@dataclass(frozen=True)
class FireObservation:
    longitude: float
    latitude: float
    kind: str
    status: str = ""
    name: str = ""
    identifier: str = ""
    size_hectares: float | None = None
    fire_of_note: bool = False
    detection_count: int = 1
    frp_mw: float | None = None
    observed_at: str = ""


@dataclass(frozen=True)
class FireActivity:
    source: str
    retrieved_at: dt.datetime
    observations: tuple[FireObservation, ...]
    cached: bool = False
    stale: bool = False

    @property
    def is_active_fire_feed(self) -> bool:
        return self.source == "bcws_active_fires"

    @property
    def footer_label(self) -> str:
        if self.is_active_fire_feed:
            prefix = "BCWS active fires"
        else:
            prefix = "24-h satellite hotspot clusters: orange squares"
        if self.stale:
            prefix += " (cached)"
        return prefix


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _feature_coordinates(feature: dict[str, object]) -> tuple[float, float] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    try:
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return None
    west, east, south, north = REQUEST_EXTENT
    if not (west <= longitude <= east and south <= latitude <= north):
        return None
    return longitude, latitude


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_active_fire_features(features: list[dict[str, object]]) -> tuple[FireObservation, ...]:
    observations: list[FireObservation] = []
    for feature in features:
        coordinates = _feature_coordinates(feature)
        properties = feature.get("properties")
        if coordinates is None or not isinstance(properties, dict):
            continue
        status = str(properties.get("FIRE_STATUS") or "").strip()
        if status.casefold() == "out":
            continue
        identifier = str(properties.get("FIRE_NUMBER") or "").strip()
        name = str(properties.get("INCIDENT_NAME") or "").strip()
        if name.casefold() == identifier.casefold():
            name = ""
        fire_of_note = (
            str(properties.get("FIRE_OF_NOTE_IND") or "").strip().upper() == "Y"
            or status.casefold() == "fire of note"
        )
        observations.append(
            FireObservation(
                longitude=coordinates[0],
                latitude=coordinates[1],
                kind="active_fire",
                status=status,
                name=name,
                identifier=identifier,
                size_hectares=_optional_float(properties.get("CURRENT_SIZE")),
                fire_of_note=fire_of_note,
            )
        )
    return tuple(observations)


def download_active_fires(now: dt.datetime | None = None) -> FireActivity:
    features: list[dict[str, object]] = []
    offset = 0
    page_size = 1000
    while True:
        response = requests.get(
            ACTIVE_FIRE_URL,
            params={
                "where": "FIRE_STATUS <> 'Out'",
                "outFields": (
                    "LATITUDE,LONGITUDE,CURRENT_SIZE,FIRE_NUMBER,FIRE_STATUS,"
                    "FIRE_CAUSE,FIRE_OF_NOTE_IND,INCIDENT_NAME"
                ),
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "geojson",
            },
            timeout=(15, 60),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "error" in payload:
            raise RuntimeError("BCWS active-fire service returned an invalid response.")
        page = payload.get("features")
        if not isinstance(page, list):
            raise RuntimeError("BCWS active-fire response did not contain GeoJSON features.")
        features.extend(item for item in page if isinstance(item, dict))
        if len(page) < page_size:
            break
        offset += len(page)
    return FireActivity(
        source="bcws_active_fires",
        retrieved_at=now or utc_now(),
        observations=parse_active_fire_features(features),
    )


def cluster_hotspot_features(features: list[dict[str, object]]) -> tuple[FireObservation, ...]:
    west, _, south, _ = REQUEST_EXTENT
    longitude_step, latitude_step = HOTSPOT_GRID_DEGREES
    buckets: dict[tuple[int, int], list[tuple[float, float, float, str]]] = {}
    for feature in features:
        coordinates = _feature_coordinates(feature)
        properties = feature.get("properties")
        if coordinates is None or not isinstance(properties, dict):
            continue
        longitude, latitude = coordinates
        frp = max(0.1, _optional_float(properties.get("frp")) or 0.1)
        observed_at = str(properties.get("rep_date") or "")
        key = (
            int(math.floor((longitude - west) / longitude_step)),
            int(math.floor((latitude - south) / latitude_step)),
        )
        buckets.setdefault(key, []).append((longitude, latitude, frp, observed_at))

    observations: list[FireObservation] = []
    for values in buckets.values():
        total_weight = sum(value[2] for value in values)
        longitude = sum(value[0] * value[2] for value in values) / total_weight
        latitude = sum(value[1] * value[2] for value in values) / total_weight
        observations.append(
            FireObservation(
                longitude=longitude,
                latitude=latitude,
                kind="hotspot",
                status="Satellite hotspot",
                detection_count=len(values),
                frp_mw=sum(value[2] for value in values),
                observed_at=max((value[3] for value in values), default=""),
            )
        )
    return tuple(observations)


def download_hotspots(now: dt.datetime | None = None) -> FireActivity:
    west, east, south, north = REQUEST_EXTENT
    response = requests.get(
        HOTSPOT_URL,
        params={
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": "public:hotspots_last24hrs",
            "outputFormat": "application/json",
            "srsName": "CRS:84",
            "propertyName": "geometry,lat,lon,rep_date,frp,sensor,satellite,agency",
            "CQL_FILTER": (
                f"agency = 'BC' AND lon BETWEEN {west:g} AND {east:g} "
                f"AND lat BETWEEN {south:g} AND {north:g}"
            ),
            "maxFeatures": 100000,
        },
        timeout=(15, 120),
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise RuntimeError("CWFIS hotspot response did not contain GeoJSON features.")
    return FireActivity(
        source="cwfis_hotspots_24h",
        retrieved_at=now or utc_now(),
        observations=cluster_hotspot_features(
            [feature for feature in features if isinstance(feature, dict)]
        ),
    )


def write_cache(activity: FireActivity, path: Path) -> None:
    payload = {
        "source": activity.source,
        "retrieved_at": activity.retrieved_at.astimezone(dt.timezone.utc).isoformat(),
        "observations": [asdict(observation) for observation in activity.observations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(path)


def read_cache(path: Path) -> FireActivity | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        retrieved_at = dt.datetime.fromisoformat(payload["retrieved_at"])
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=dt.timezone.utc)
        observations = tuple(FireObservation(**item) for item in payload["observations"])
        source = str(payload["source"])
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if source not in {"bcws_active_fires", "cwfis_hotspots_24h"}:
        return None
    return FireActivity(source, retrieved_at, observations, cached=True)


def _fresh(activity: FireActivity | None, now: dt.datetime, maximum_age: dt.timedelta) -> bool:
    if activity is None:
        return False
    return dt.timedelta(0) <= now - activity.retrieved_at <= maximum_age


def load_fire_activity(
    active_cache: Path = ACTIVE_FIRE_CACHE,
    hotspot_cache: Path = HOTSPOT_CACHE,
    *,
    now: dt.datetime | None = None,
    logger: Callable[[str], None] | None = None,
) -> FireActivity | None:
    """Load current incidents, falling back to hotspots without breaking plotting."""
    now = now or utc_now()
    cached_active = read_cache(active_cache)
    if _fresh(cached_active, now, ACTIVE_CACHE_MAX_AGE):
        return cached_active

    try:
        activity = download_active_fires(now)
        try:
            write_cache(activity, active_cache)
        except OSError as exc:
            if logger:
                logger(f"Could not update the BCWS active-fire cache ({exc}); using live observations.")
        if logger:
            logger(f"Loaded {len(activity.observations)} current BCWS active fires.")
        return activity
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        if logger:
            logger(f"BCWS active-fire feed unavailable ({exc}); trying CWFIS hotspots.")

    cached_hotspots = read_cache(hotspot_cache)
    if _fresh(cached_hotspots, now, HOTSPOT_CACHE_MAX_AGE):
        return cached_hotspots
    try:
        activity = download_hotspots(now)
        try:
            write_cache(activity, hotspot_cache)
        except OSError as exc:
            if logger:
                logger(f"Could not update the CWFIS hotspot cache ({exc}); using live observations.")
        if logger:
            logger(f"Loaded {len(activity.observations)} CWFIS 24-h hotspot clusters.")
        return activity
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        if logger:
            logger(f"CWFIS hotspot feed unavailable ({exc}).")

    for cached in (cached_active, cached_hotspots):
        if _fresh(cached, now, STALE_CACHE_MAX_AGE):
            if logger:
                logger(f"Using cached {cached.source} observations from {cached.retrieved_at:%H%MZ}.")
            return replace(cached, stale=True)
    if logger:
        logger("No current fire-activity overlay is available; continuing without it.")
    return None
