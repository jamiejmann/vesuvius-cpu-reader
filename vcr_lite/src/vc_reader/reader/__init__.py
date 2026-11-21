"""zarr-python backed replacements for vc_reader.reader.open_node / ZarrWriter."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
import numpy as np
import zarr

__all__ = ["open_node", "load_basic_auth", "ZarrWriter"]

def _https(url: str) -> str:
    # s3://bucket/key -> global virtual-host endpoint (anonymous, works for the open-data bucket)
    if url.startswith("s3://"):
        bucket, _, key = url[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url

def load_basic_auth(path: str | os.PathLike[str]) -> Any:
    data = json.loads(Path(path).read_text())
    return (data.get("username") or data.get("user"), data.get("password"))

def open_node(path: str, auth: Any = "auto", cache_dir: str | None = None,
              cache_max_bytes: int | None = None, **_: Any):
    """Open a Zarr v2/v3 array or group, local or remote (http(s)/s3), read-only."""
    path = str(path)
    if "://" in path:
        url = _https(path).rstrip("/")
        opts: dict[str, Any] = {}
        if isinstance(auth, tuple) and auth[0]:
            import aiohttp
            opts["client_kwargs"] = {"auth": aiohttp.BasicAuth(auth[0], auth[1] or "")}
        if cache_dir:  # simple on-disk chunk cache via fsspec's filecache
            store = zarr.storage.FsspecStore.from_url(
                "filecache::" + url, read_only=True,
                storage_options={"filecache": {"cache_storage": str(cache_dir)}, "https": opts})
        else:
            store = zarr.storage.FsspecStore.from_url(url, read_only=True, storage_options=opts)
        return zarr.open(store=store, mode="r")
    return zarr.open(store=path, mode="r")

class ZarrWriter:
    """Only the `create` entry point used for the float32 accumulation arrays."""
    @staticmethod
    def create(path, *, shape, chunks, dtype, overwrite: bool = False, auth: Any = None,
               fill_value: Any = 0, **_: Any):
        return zarr.create_array(store=str(path), shape=tuple(shape), chunks=tuple(chunks),
                                 dtype=np.dtype(dtype), fill_value=fill_value,
                                 overwrite=overwrite, zarr_format=2)
