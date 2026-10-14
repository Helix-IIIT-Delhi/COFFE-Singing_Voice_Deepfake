"""COFFE: Chernoff Distance Fusion for Singing Voice Deepfake Source Attribution"""

__version__ = "1.0.0"
__author__ = "Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, and others"

from .fusion import COFFEFusion
from .distance import chernoff_distance

__all__ = ["COFFEFusion", "chernoff_distance"]
