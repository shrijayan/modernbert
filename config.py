# config.py
from dataclasses import dataclass
from enum import Enum

class ClassificationType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

@dataclass
class ModelConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 8192
    batch_size: int = 1
    num_epochs: int = 0.001
    seed: int = 42
    classification_type: ClassificationType = ClassificationType.MULTICLASS

