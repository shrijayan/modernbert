import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List

class RedditPostClassifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = None
        self.labels = None
        # Add severity mapping based on the training data
        self.severity_mapping = {
            'low': 0,
            'medium': 1,
            'high': 2
        }
        self.reverse_severity_mapping = {v: k for k, v in self.severity_mapping.items()}
        
    def load_labels_from_dataset(self, dataset_path: str) -> List[str]:
        unique_labels = set()
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        for entry in data:
            if 'parsed_data' in entry:
                cats = entry['parsed_data'].get('severity', [])
                unique_labels.update([cats] if isinstance(cats, str) else cats)
        
        self.labels = sorted(list(unique_labels))
        return self.labels
    
    def initialize_model(self, model_path: str = None):
        if not self.labels:
            raise ValueError("Labels must be loaded before initializing model")
        
        path = model_path or self.tokenizer.name_or_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=len(self.labels),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        if not self.model or not self.labels:
            raise ValueError("Model and labels must be initialized before prediction")
            
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get the predicted label with highest probability
            predicted_idx = torch.argmax(probs, dim=-1).item()
            predicted_label = self.labels[predicted_idx]
            confidence = float(probs[0][predicted_idx])
            
            return {
                "labels": [predicted_label],
                "probabilities": [f"{confidence:.2%}"]
            }

def process_reddit_post(classifier: RedditPostClassifier, post_data: Dict) -> Dict:
    text = post_data["raw_data"]["text"]
    predictions = classifier.predict(text)
    
    return {
        "predictions": {
            "categories": predictions["labels"],
            "confidence_scores": predictions["probabilities"]
        }
    }

if __name__ == "__main__":
    # Key modifications:
    # 1. Changed to single-label classification
    # 2. Using softmax instead of sigmoid
    # 3. Added severity mapping
    classifier = RedditPostClassifier("severity_model_final")
    
    classifier.load_labels_from_dataset("dataset/medical_mimic(1).json")
    classifier.initialize_model()
    
    with open("dataset/medical_mimic(1).json", 'r') as f:
        test_data = json.load(f)[0]
    
    result = process_reddit_post(classifier, test_data)
    print(json.dumps(result, indent=2))