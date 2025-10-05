# src/utils/zarrio.py
from __future__ import annotations
import zarr

def create_array(path: str, name: str, shape, chunks, dtype="f2", level: int = 3):
    grp = zarr.open(path, mode="a")  # append (do not overwrite other arrays)
    # if the array exists, drop it to recreate with new shape/dtype
    try:
        if name in grp:
            del grp[name]
    except Exception:
        pass
    try:
        from zarr.codecs import Zstd as ZstdCodec
        return grp.create_array(name, shape=shape, chunks=chunks, dtype=dtype, codecs=[ZstdCodec(level=level)])
    except Exception:
        import numcodecs
        return grp.create(name, shape=shape, chunks=chunks, dtype=dtype, compressor=numcodecs.Zstd(level=level))
