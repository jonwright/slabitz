"""GPU zstd decompressor compiled from Slang to CUDA.

The kernels are authored in Slang (transcribed from Microsoft's DirectStorage
HLSL decoder, MIT) and lowered to CUDA through slangc + nvcc, then loaded with
the CUDA Driver API via :mod:`slabitz.loader`.  Device arrays are exchanged
zero-copy with PyTorch through ``__cuda_array_interface__`` / DLPack.
"""

from . import loader

__all__ = ["loader"]
