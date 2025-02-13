# data_processor.py
import json
from typing import Dict, List, Tuple
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def load_data(self, file1: str, file2: str) -> Tuple[Tuple, Tuple]:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        data = data1 + data2
        
        severity_data = (
            [entry['raw_data']['text'] for entry in data],
            [self.severity_mapping[entry['parsed_data']['severity']] for entry in data]
        )
        
        action_data = (
            [entry['raw_data']['text'] for entry in data],
            [int(entry['parsed_data']['immediate_action_required']) for entry in data]
        )
        
        return severity_data, action_data

    def prepare_dataset(self, texts: List[str], labels: List) -> HFDataset:
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return HFDataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })

    def split_data(self, texts: List[str], labels: List) -> Tuple:
        return train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )