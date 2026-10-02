"""High-resolution feature fusion with LocNet-inspired boundary refinement."""

from .config import ModelConfig
from .model import HSIBoundaryDetector
from .losses import RefinementCriterion

__all__ = ["ModelConfig", "HSIBoundaryDetector", "RefinementCriterion"]
