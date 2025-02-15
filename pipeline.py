# pipeline.py
import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer
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

    def train_model(self, dataset_splits: dict, model_name: str) -> None:
        # Unpack the splits
        train_texts, train_labels = dataset_splits['train']
        val_texts, val_labels = dataset_splits['validation']
        test_texts, test_labels = dataset_splits['test']
        
        # Prepare datasets for training and validation
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
        
        # Save the final model and tokenizer
        trainer.save_model(f"./{model_name}_final")
        self.tokenizer.save_pretrained(f"./{model_name}_final")
        
        # Optional: Evaluate on test set
        test_dataset = self.data_processor.prepare_dataset(test_texts, test_labels)
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test Results: {test_results}")

    def run(self, dataset_name: str = "shrijayan/medical_mimic"):
        # Load data from Hugging Face dataset
        dataset_splits = self.data_processor.load_data(dataset_name)
        
        # Train the model
        self.train_model(dataset_splits, "multilabel_model")