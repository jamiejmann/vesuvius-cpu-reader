"""NumPy port of vc_reader's CPU Normalize (unmasked path), matching
vc_reader/src/cpp/transforms/cpu/normalization.hpp: NumPy linear percentiles,
float64 statistics, results cast back to the input float dtype."""
from __future__ import annotations
import numpy as np

__all__ = ["Normalize"]

def _robust_scale(mad: float, deviation: float, span: float) -> float:
    s = 1.4826 * mad
    if np.isfinite(s) and s >= 1e-6: return s
    if np.isfinite(deviation) and abs(deviation) >= 1e-6: return abs(deviation)
    if np.isfinite(span) and abs(span) / 2 >= 1e-6: return abs(span) / 2
    return 1.0

class Normalize:
    def __init__(self, mode: str = "zscore", *, nonzero: bool = True, eps: float = 0.0,
                 lower: float = 1.0, upper: float = 99.0, clip: bool = True, median: bool = False,
                 center: float = 0.0, scale: float = 1.0, bounds=None, low=None, high=None, **kw):
        if kw.get("masked"):
            raise NotImplementedError("vc-reader-lite: masked normalization is not supported")
        self.mode, self.nonzero, self.eps = mode, nonzero, eps
        self.lower, self.upper, self.clip, self.median = lower, upper, clip, median
        self.center, self.scale = center, scale
        lo, hi = (bounds if bounds is not None else (low, high))
        self.low = -np.inf if lo is None else float(lo)
        self.high = np.inf if hi is None else float(hi)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image)
        out = np.empty_like(x, dtype=x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
        # vc_reader treats axis 0 as batch, axis 1 as channel; stats per (batch, channel)
        flat_in = x.reshape(x.shape[0] * x.shape[1], -1) if x.ndim >= 2 else x.reshape(1, -1)
        flat_out = out.reshape(flat_in.shape)
        for i in range(flat_in.shape[0]):
            flat_out[i] = self._channel(flat_in[i].astype(np.float64))
        return out

    def _channel(self, v: np.ndarray) -> np.ndarray:
        m = self.mode
        if not np.all(np.isfinite(v)):
            raise ValueError("normalization requires finite crop values")
        if m == "fixed":
            r = (np.clip(v, self.low, self.high) - self.center) / self.scale
        elif m in ("robust", "percentile"):
            sel = v
            deviation = float(np.sqrt(np.mean((sel - sel.mean()) ** 2))) if m == "robust" else np.nan
            low, high = float(sel.min()), float(sel.max())
            if m == "percentile" or self.clip:
                low, high = (float(np.percentile(sel, self.lower)), float(np.percentile(sel, self.upper)))
                v = np.clip(v, low, high); sel = v
            if m == "percentile":
                s = 0.5 * (high - low) if self.median else high - low
                s = s if (np.isfinite(s) and s >= 1e-6) else (1.0 if self.median else 0.0)
                if s == 0: return np.zeros_like(v)
                c = float(np.percentile(sel, 50)) if self.median else low
            else:
                c = float(np.percentile(sel, 50))
                s = _robust_scale(float(np.percentile(np.abs(sel - c), 50)), deviation, high - low)
            r = (v - c) / s
        elif m == "minmax":
            lo, hi = v.min(), v.max(); r = (v - lo) / max(hi - lo, self.eps or 1e-12)
        else:  # zscore
            sel = v[v != 0] if self.nonzero else v
            if sel.size == 0: return v
            mu, sd = sel.mean(), max(sel.std(), self.eps)
            r = (v - mu) / (sd if sd > 0 else 1.0)
            if self.nonzero: r[v == 0] = 0
        r[~np.isfinite(r)] = 0
        return r
