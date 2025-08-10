# src/utils/zarrio.py
from __future__ import annotations
import os
import zarr

def create_array(path: str, name: str, shape, chunks, dtype="f2", level: int = 3):
    """
    Create a Zarr array at `path` with dataset `name`, working on both Zarr v2 and v3.
    Returns the created array.
    """
    grp = zarr.open(path, mode="a")  # append, don't overwrite
    try:
        if name in grp:              # ensure fresh create with correct shape
            del grp[name]
    except Exception:
        pass
    # Try v3 API 
    try:
        from zarr.codecs import Zstd as ZstdCodec  # v3
        return grp.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            codecs=[ZstdCodec(level=level)],
        )
    except Exception:
        # Fallback: v2 API
        import numcodecs
        return grp.create(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=numcodecs.Zstd(level=level),
        )
