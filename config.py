# config.py
from dataclasses import dataclass
from enum import Enum

class ClassificationType(Enum):
    BINARY = "binary"
    MULTICLASS = "multi_label_classification"
    MULTILABEL = "multilabel"

@dataclass
class ModelConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 3
    seed: int = 42
    classification_type: ClassificationType = ClassificationType.MULTILABEL
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    fp16: bool = True
    early_stopping_patience: int = 5
    eval_and_save_steps: int = 500
    eval_and_save_strategy: str = "steps"
    logging_steps: int = 100
    dataset_name: str = "shrijayan/medical_mimic"