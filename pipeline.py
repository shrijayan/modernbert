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
        
        # Enable TF32 for better performance on Ampere GPUs
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            
            # Additional optimization: enable cudnn deterministic mode since we set a seed
            torch.backends.cudnn.deterministic = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            model_max_length=config.max_length
        )
        
        self.data_processor = DataProcessor(self.tokenizer, config.max_length)
        
        self.model_trainer = ModelTrainer()

    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def run(self, dataset_name):
        dataset_splits = self.data_processor.load_data(dataset_name)
        
        train_texts, train_labels = dataset_splits['train']
        val_texts, val_labels = dataset_splits['validation']
        test_texts, test_labels = dataset_splits['test']
        
        train_dataset = self.data_processor.prepare_dataset(train_texts, train_labels)
        val_dataset = self.data_processor.prepare_dataset(val_texts, val_labels)
        
        model_name = "multilabel_model"
        trainer = self.model_trainer.train(
            model_name=self.config.model_name,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            num_labels=5,
            output_dir=f"./{model_name}",
            config=self.config,
            problem_type=self.config.classification_type
        )
        
        # Save the final model and tokenizer
        trainer.save_model(f"./{model_name}_final")
        self.tokenizer.save_pretrained(f"./{model_name}_final")
        
        # Evaluate on test set
        test_dataset = self.data_processor.prepare_dataset(test_texts, test_labels)
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test Results: {test_results}")
