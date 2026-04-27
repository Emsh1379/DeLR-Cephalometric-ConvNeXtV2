"""
Dual-encoder Landmark Regression (DeLR) package.

This package bundles:
  - The DeLR/ D-CeLR model architecture (`model.build_dcelr`).
  - Dataset helpers tailored for the Aariz cephalometric dataset.
  - Metric utilities for mean radial error (MRE) and success detection rates (SDR).
"""

from .model import DCeLR, build_dcelr

__all__ = ["DCeLR", "build_dcelr"]
