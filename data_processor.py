# data_processor.py
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def create_multilabel(self, severity: str, action: bool) -> List[int]:
        # Convert severity to one-hot encoding
        severity_idx = self.severity_mapping[severity]
        severity_one_hot = [1 if i == severity_idx else 0 for i in range(4)]
        # Add action as the last element
        return severity_one_hot + [int(action)]

    def load_data(self, file1: str, file2: str) -> Tuple[List[str], List[List[int]]]:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        data = data1 + data2
        
        texts = [entry['raw_data']['text'] for entry in data]
        # Create multilabel data with one-hot encoded severity and binary action
        labels = [
            self.create_multilabel(
                entry['parsed_data']['severity'],
                entry['parsed_data']['immediate_action_required']
            )
            for entry in data
        ]
        
        return texts, labels

    def split_data(self, texts: List[str], labels: List[List[int]]) -> Tuple:
        """Split the data into training and validation sets."""
        return train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42
        )
        
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