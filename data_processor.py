# data_processor.py
import torch
from typing import List, Tuple
from datasets import load_dataset, Dataset as HFDataset

class DataProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def create_multilabel(self, severity: str, action: bool) -> List[int]:
        severity_idx = self.severity_mapping[severity]
        severity_one_hot = [1 if i == severity_idx else 0 for i in range(4)]
        return severity_one_hot + [int(action)]

    def load_data(self, dataset_name) -> Tuple[dict]:
        dataset = load_dataset(dataset_name)
        
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