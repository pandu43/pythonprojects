"""Compatibility shim package exposing TRNG modules from `src.trng`.

These wrappers keep existing import paths (`import TRNG.PUF_RNG`) working
while the real implementation lives under `src/trng/`.
"""

__all__ = ["PUF_RNG", "puf_adapter", "PUFDesign_0"]
