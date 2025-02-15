# performance_debug.py
import torch
import psutil
import os
import logging
from typing import Optional, Dict
import time
from dataclasses import dataclass
from transformers import Trainer

@dataclass
class PerformanceMetrics:
    gpu_memory_allocated: float
    gpu_memory_cached: float
    cpu_memory_used: float
    batch_processing_time: float
    cuda_memory_summary: Optional[Dict] = None

class PerformanceDebugTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_logger = logging.getLogger('performance_debug')
        handler = logging.FileHandler('performance_debug.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.debug_logger.addHandler(handler)
        self.debug_logger.setLevel(logging.INFO)
        
    def training_step(self, model, inputs):
        start_time = time.time()
        metrics = self._get_performance_metrics()
        self.debug_logger.info(f"Before training step:\n{self._format_metrics(metrics)}")
        
        loss = super().training_step(model, inputs)
        
        metrics = self._get_performance_metrics()
        batch_time = time.time() - start_time
        metrics.batch_processing_time = batch_time
        self.debug_logger.info(f"After training step:\n{self._format_metrics(metrics)}")
        
        if hasattr(torch.cuda, 'memory_summary'):
            self.debug_logger.info(f"CUDA Memory Summary:\n{torch.cuda.memory_summary()}")
        
        return loss

    def _get_performance_metrics(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics(
            gpu_memory_allocated=torch.cuda.memory_allocated() / 1024**2,
            gpu_memory_cached=torch.cuda.memory_reserved() / 1024**2,
            cpu_memory_used=psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            batch_processing_time=0.0
        )
        
        if hasattr(torch.cuda, 'memory_summary'):
            metrics.cuda_memory_summary = str(torch.cuda.memory_summary())
        
        return metrics
    
    def _format_metrics(self, metrics: PerformanceMetrics) -> str:
        return f"""
Performance Metrics:
- GPU Memory Allocated: {metrics.gpu_memory_allocated:.2f} MB
- GPU Memory Cached: {metrics.gpu_memory_cached:.2f} MB
- CPU Memory Used: {metrics.cpu_memory_used:.2f} MB
- Batch Processing Time: {metrics.batch_processing_time:.2f} seconds
"""

def optimize_batch_size(trainer: Trainer, start_size: int = 1, max_size: int = 512) -> int:
    """Find the optimal batch size through binary search."""
    current_size = start_size
    step_size = 2
    
    while current_size <= max_size:
        try:
            # Try to process a batch
            trainer.args.per_device_train_batch_size = current_size
            trainer.args.per_device_eval_batch_size = current_size
            
            # Get a batch from the dataloader
            batch = next(iter(trainer.get_train_dataloader()))
            
            # Try to process it
            trainer.training_step(trainer.model, batch)
            
            logging.info(f"Successfully processed batch size: {current_size}")
            current_size *= step_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If we hit OOM, return the last successful batch size
                return current_size // step_size
            raise e
    
    return current_size // step_size