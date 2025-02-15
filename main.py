# main.py
import os
import logging
import torch
from datetime import datetime
from pathlib import Path
from config import ModelConfig, ClassificationType
from pipeline import Pipeline
from model_push import push_trained_model
from performance_debug import PerformanceDebugTrainer, optimize_batch_size
from transformers import TrainingArguments
from model_trainer import compute_multilabel_metrics
from transformers import AutoModelForSequenceClassification


def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/training_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

def get_training_arguments(config: ModelConfig, output_dir: str) -> TrainingArguments:
    """Create training arguments with optimal settings"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        greater_is_better=True,
        save_total_limit=2,  # Keep only the last 2 checkpoints
        report_to="tensorboard",
        dataloader_num_workers=4,  # Adjust based on CPU cores
        dataloader_pin_memory=True,  # Better GPU memory transfer
        group_by_length=True,  # Reduce padding by grouping similar lengths
    )

def log_system_info():
    """Log system information for debugging"""
    logging.info("=== System Information ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU model: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logging.info("========================")

def main():
    # Set up logging
    setup_logging()
    log_system_info()
    
    # Initialize configuration
    config = ModelConfig()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./models/medical_classifier_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize pipeline
        pipeline = Pipeline(config)
        
        # Load datasets
        logging.info("Loading datasets...")
        dataset_splits = pipeline.data_processor.load_data()
        
        # Prepare datasets
        train_dataset = pipeline.data_processor.prepare_dataset(
            dataset_splits['train'][0], 
            dataset_splits['train'][1]
        )
        eval_dataset = pipeline.data_processor.prepare_dataset(
            dataset_splits['validation'][0], 
            dataset_splits['validation'][1]
        )
        test_dataset = pipeline.data_processor.prepare_dataset(
            dataset_splits['test'][0], 
            dataset_splits['test'][1]
        )
        
        # Get training arguments
        training_args = get_training_arguments(config, output_dir)
        
        # Initialize trainer with performance debugging
        trainer = PerformanceDebugTrainer(
            args=training_args,
            model=AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=5,
                problem_type="multi_label_classification"
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_multilabel_metrics
        )
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cleared")

if __name__ == "__main__":
    main()