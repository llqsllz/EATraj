"""WGS84 engineering conversion and historical estimators; no future inputs."""
from __future__ import annotations

import numpy as np


def geodetic_to_enu(llh: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """lon/lat degrees, height metres -> a fixed shared ENU frame.

    WGS84 and altitude datum are caller-declared assumptions, not established here.
    """
    llh = np.asarray(llh, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    def ecef(x):
        lon, lat, h = np.moveaxis(x, -1, 0)
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        a, e2 = 6378137.0, 6.6943799901413165e-3
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        return np.stack(((n + h) * np.cos(lat) * np.cos(lon),
                         (n + h) * np.cos(lat) * np.sin(lon),
                         (n * (1 - e2) + h) * np.sin(lat)), axis=-1)
    lon, lat = np.deg2rad(origin[:2])
    r = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                  [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return (ecef(llh) - ecef(origin)) @ r.T


def trailing_velocity(times: np.ndarray, positions: np.ndarray, window_s: float = 1.,
                      min_samples: int = 3, min_span_s: float = .4) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares slopes restricted to the supplied history; no implicit lookback."""
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (len(times), 3) or np.any(np.diff(times) <= 0):
        raise ValueError("Expected strictly increasing times and aligned [T,3] positions")
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError("Velocity estimator accepts valid observations only")
    # Shift origin to reduce cancellation for absolute replay timestamps.
    t = times - times[0]
    lo = np.searchsorted(t, t - window_s - 1e-8)
    hi = np.arange(len(t)) + 1
    n = hi - lo
    def sums(x):
        c = np.concatenate((np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)))
        return c[hi] - c[lo]
    st, st2, sp, stp = sums(t), sums(t*t), sums(positions), sums(t[:, None]*positions)
    den = st2 - st*st/n
    valid = (n >= min_samples) & ((t-t[lo]) >= min_span_s-1e-8) & (den > 1e-12)
    velocity = np.zeros_like(positions)
    velocity[valid] = (stp[valid] - st[valid, None]*sp[valid]/n[valid, None]) / den[valid, None]
    return velocity, valid


def endpoint_rotation(velocity: np.ndarray, valid: np.ndarray, min_speed: float = 1.) -> tuple[np.ndarray, bool]:
    """Local-to-shared yaw rotation from the most recent reliable historical direction."""
    good = np.flatnonzero(valid & (np.linalg.norm(velocity[:, :2], axis=-1) >= min_speed))
    if not len(good):
        return np.eye(3), False
    x, y = velocity[good[-1], :2]
    theta = np.arctan2(y, x)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]), True
