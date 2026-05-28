"""
numcompute_stream package
This package contains a tiny streaming ML framework 
"""

from .pipeline import Pipeline
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier
from .metrics import Accuracy, ConfusionMatrix
from .stream import StreamTrainer

__all__ = [
    "Pipeline",
    "StandardScaler",
    "Imputer",
    "OneHotEncoder",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "Accuracy",
    "ConfusionMatrix",
    "StreamTrainer",
]