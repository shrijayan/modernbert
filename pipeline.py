# pipeline.py
# import wandb
import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Tuple
from config import ModelConfig
from data_processor import DataProcessor
from model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.set_seed()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.data_processor = DataProcessor(self.tokenizer, config.max_length)
        self.model_trainer = ModelTrainer()

    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def train_model(self, texts: list[str], labels: list[list[int]], model_name: str) -> None:
        train_texts, val_texts, train_labels, val_labels = self.data_processor.split_data(texts, labels)
        
        train_dataset = self.data_processor.prepare_dataset(train_texts, train_labels)
        val_dataset = self.data_processor.prepare_dataset(val_texts, val_labels)
        
        logger.info(f"Training {model_name}...")
        trainer = self.model_trainer.train(
            self.config.model_name,
            train_dataset,
            val_dataset,
            num_labels=5,  # 4 severity levels + 1 action required
            output_dir=f"./{model_name}",
            config=self.config,
            problem_type="multi_label_classification"
        )
        
        trainer.save_model(f"./{model_name}_final")
        self.tokenizer.save_pretrained(f"./{model_name}_final")

    def run(self, file1: str, file2: str):
        # wandb.init(project="medical-text-classification")
        texts, labels = self.data_processor.load_data(file1, file2)
        self.train_model(texts, labels, "multilabel_model")
        # wandb.finish()