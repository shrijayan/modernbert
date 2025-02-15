# data_processor.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import os

class DataProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def create_multilabel(self, severity: str, action: bool) -> List[int]:
        # Convert severity to one-hot encoding
        severity_idx = self.severity_mapping[severity]
        severity_one_hot = [1 if i == severity_idx else 0 for i in range(4)]
        # Add action as the last element
        return severity_one_hot + [int(action)]

    def load_data(self, dataset_name: str = "shrijayan/medical_mimic") -> Tuple[dict]:
        """
        Load data from Hugging Face dataset
        """
        # Load the dataset with predefined splits
        dataset = load_dataset(dataset_name, token=os.getenv("HUGGINGFACE_TOKEN"))
        
        # Prepare data for each split
        def process_split(split_data):
            texts = [entry['raw_data']['text'] for entry in split_data]
            labels = [
                self.create_multilabel(
                    entry['parsed_data']['severity'],
                    entry['parsed_data']['immediate_action_required']
                )
                for entry in split_data
            ]
            return texts, labels
        
        # Process each split
        train_texts, train_labels = process_split(dataset['train'])
        val_texts, val_labels = process_split(dataset['validation'])
        test_texts, test_labels = process_split(dataset['test'])
        
        return {
            'train': (train_texts, train_labels),
            'validation': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
        
    def prepare_dataset(self, texts: List[str], labels: List[List[int]]) -> HFDataset:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Convert labels to torch tensor and ensure they are of type float
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        return HFDataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels_tensor
        })