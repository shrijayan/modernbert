# config.py
from dataclasses import dataclass
from enum import Enum

class ClassificationType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

@dataclass
class ModelConfig:
    model_name: str = "microsoft/deberta-base"
    max_length: int = 512
    batch_size: int = 8
    num_epochs: int = 0.01
    seed: int = 42
    classification_type: ClassificationType = ClassificationType.MULTICLASS

