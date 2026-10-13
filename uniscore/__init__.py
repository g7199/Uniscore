"""UniScore: aggregating LLM-judge criteria by fitting where supervision is plentiful and weighing by tested evidence where it is scarce."""

from .estimators import AnchorModel, EqualSigned, RankRidgeCV, SomersAHP, standardized
from .model import UniScore

__all__ = ["UniScore", "SomersAHP", "EqualSigned", "AnchorModel", "RankRidgeCV", "standardized"]
__version__ = "1.0.0"
