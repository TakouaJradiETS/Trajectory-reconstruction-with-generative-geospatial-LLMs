"""Spatial distance metrics using haversine for AIS evaluation."""

import math
import numpy as np
from typing import List, Tuple

Coords = List[Tuple[float, float]]  # list of (lat, lon)

_EARTH_R_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Flat-Earth coordinate conversion (valid for offsets < ~100 km)
# ---------------------------------------------------------------------------

def latlon_to_enu_m(
    lat: float, lon: float,
    ref_lat: float, ref_lon: float,
) -> tuple:
    """
    Flat-Earth east/north offset (metres) of (lat, lon) from (ref_lat, ref_lon).
    Returns (east_m, north_m).
    """
    north_m = math.radians(lat - ref_lat) * _EARTH_R_M
    avg_lat  = math.radians((lat + ref_lat) / 2.0)
    east_m   = math.radians(lon - ref_lon) * _EARTH_R_M * math.cos(avg_lat)
    return east_m, north_m


def enu_to_latlon(
    east_m: float, north_m: float,
    ref_lat: float, ref_lon: float,
) -> tuple:
    """
    Inverse of latlon_to_enu_m.  Returns (lat, lon) in degrees.
    """
    dlat = math.degrees(north_m / _EARTH_R_M)
    avg_lat = math.radians(ref_lat + dlat / 2.0)
    cos_avg = math.cos(avg_lat) or 1e-9
    dlon = math.degrees(east_m / (_EARTH_R_M * cos_avg))
    return ref_lat + dlat, ref_lon + dlon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _pairwise_haversine(pred: Coords, true: Coords) -> List[float]:
    n = min(len(pred), len(true))
    return [haversine_km(true[i][0], true[i][1], pred[i][0], pred[i][1]) for i in range(n)]


def compute_mse_km2(pred: Coords, true: Coords) -> float:
    dists = _pairwise_haversine(pred, true)
    return float(np.mean([d ** 2 for d in dists])) if dists else float("nan")


def compute_mae_km(pred: Coords, true: Coords) -> float:
    dists = _pairwise_haversine(pred, true)
    return float(np.mean(dists)) if dists else float("nan")


def compute_ade_km(pred: Coords, true: Coords) -> float:
    """Average Displacement Error across all aligned steps."""
    return compute_mae_km(pred, true)


def compute_fde_km(pred: Coords, true: Coords) -> float:
    """Final Displacement Error at the last aligned step."""
    n = min(len(pred), len(true))
    if n == 0:
        return float("nan")
    return haversine_km(true[n - 1][0], true[n - 1][1], pred[n - 1][0], pred[n - 1][1])


def compute_dtw_km(pred: Coords, true: Coords) -> float:
    """DTW distance between two coordinate sequences using haversine."""
    n, m = len(pred), len(true)
    if n == 0 or m == 0:
        return float("nan")
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = haversine_km(true[j - 1][0], true[j - 1][1], pred[i - 1][0], pred[i - 1][1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])
