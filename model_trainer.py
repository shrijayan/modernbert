# model_trainer.py
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from config import ModelConfig
from model_utils import compute_multilabel_metrics

class ModelTrainer:
    def train(
        self,
        model_name: str,
        train_dataset: HFDataset,
        eval_dataset: HFDataset,
        num_labels: int,
        output_dir: str,
        config: ModelConfig,
        problem_type: str = "multi_label_classification"
    ) -> Trainer:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="avg_f1",
            fp16=config.fp16
        )
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            reference_compile=False,
            num_labels=num_labels,  # Explicitly pass the number of labels
            problem_type=problem_type  # Specify multi-label classification
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_multilabel_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        return trainer
