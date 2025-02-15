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
    batch_size: int = 16  # Changed from 1 to a more reasonable default
    num_epochs: int = 3   # Changed from 0.001 to a reasonable number of epochs
    seed: int = 42
    classification_type: ClassificationType = ClassificationType.MULTILABEL
    # Add missing fields referenced in main.py
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    fp16: bool = True
    dataset_name: str = "shrijayan/medical_mimic"

