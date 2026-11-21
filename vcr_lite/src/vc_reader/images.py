"""tifffile-backed replacements for vc_reader.images.read / TiffWriter."""
from __future__ import annotations
import numpy as np
import tifffile

__all__ = ["read", "TiffWriter"]

def read(path):
    return tifffile.imread(str(path))

class TiffWriter:
    """Collects tiles in a (memory-mapped if large) array, writes a tiled TIFF on close."""
    def __init__(self, path, shape, dtype, compression: str | None = "lzw", tile=(256, 256), **_):
        self.path, self.shape, self.dtype = str(path), tuple(int(s) for s in shape), np.dtype(dtype)
        self.compression, self.tile = compression, tuple(int(t) for t in tile)
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        if nbytes > 512 * 2**20:
            import tempfile
            self._tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
            self.data = np.memmap(self._tmp.name, dtype=self.dtype, mode="w+", shape=self.shape)
        else:
            self._tmp, self.data = None, np.zeros(self.shape, self.dtype)
    def write_tile(self, y0: int, x0: int, tile) -> None:
        t = np.asarray(tile, dtype=self.dtype)
        self.data[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    def __enter__(self): return self
    def __exit__(self, *exc):
        if exc[0] is None:
            tifffile.imwrite(self.path, self.data, tile=self.tile, compression=self.compression,
                             bigtiff=self.data.nbytes > 2**31)
        if self._tmp is not None:
            import os; del self.data; self._tmp.close(); os.unlink(self._tmp.name)
        return False
