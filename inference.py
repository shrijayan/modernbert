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
        
    def load_labels_from_dataset(self, dataset_path: str) -> List[str]:
        unique_labels = set()
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        for entry in data:
            if 'parsed_data' in entry:
                cats = entry['parsed_data'].get('severity', [])
                other_cats = entry['parsed_data'].get('other_cats', [])
                unique_labels.update(cats)
                unique_labels.update(other_cats)
        
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

    def predict(self, text: str, threshold: float = 0.3) -> Dict:
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
            probs = torch.sigmoid(logits)
            
            # Get top K predictions regardless of threshold
            k = min(5, len(self.labels))
            top_k_probs, top_k_indices = torch.topk(probs[0], k)
            
            pred_tuples = [(self.labels[idx], float(prob)) 
                          for idx, prob in zip(top_k_indices.tolist(), top_k_probs.tolist())
                          if float(prob) > threshold]
            
            if not pred_tuples:  # If no predictions above threshold, take top prediction
                idx = top_k_indices[0].item()
                prob = float(top_k_probs[0].item())
                pred_tuples = [(self.labels[idx], prob)]
            
            pred_labels = [label for label, _ in pred_tuples]
            pred_probs = [f"{prob:.2%}" for _, prob in pred_tuples]
            
            return {
                "labels": pred_labels,
                "probabilities": pred_probs
            }

def process_reddit_post(classifier: RedditPostClassifier, post_data: Dict) -> Dict:
    text = post_data["raw_data"]["text"]
    predictions = classifier.predict(text)
    
    return {
        "raw_data": post_data["raw_data"],
        "predictions": {
            "categories": predictions["labels"],
            "confidence_scores": predictions["probabilities"]
        }
    }

if __name__ == "__main__":
    classifier = RedditPostClassifier("severity_model_final")
    
    classifier.load_labels_from_dataset("dataset/medical_mimic(1).json")
    classifier.initialize_model()
    
    with open("dataset/medical_mimic(1).json", 'r') as f:
        test_data = json.load(f)[0]
    
    result = process_reddit_post(classifier, test_data)
    print(json.dumps(result, indent=2))