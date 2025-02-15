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
from model_utils import compute_metrics, compute_multilabel_metrics

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
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="avg_f1",  # Changed from "f1" to "avg_f1" to match compute_multilabel_metrics
            fp16=True  # Enable mixed precision training
        )
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
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