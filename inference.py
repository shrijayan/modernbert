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
        
        # Updated severity and action labels
        self.severity_labels = ['low', 'medium', 'high', 'critical']
        self.action_labels = ['no_action', 'action_required']
        
    def load_labels_from_dataset(self, dataset_path: str) -> tuple[List[str], List[str]]:
        """
        Load labels from the dataset. 
        For multi-label, we know the structure from the data processing.
        """
        return self.severity_labels, self.action_labels
    
    def initialize_model(self, model_path: str = None):
        path = model_path or self.tokenizer.name_or_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=5,  # 4 severity levels + 1 action
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        if not self.model:
            self.initialize_model()
            
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
            
            # Apply sigmoid for multi-label classification
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Separate severity and action predictions
            severity_probs = probs[:4]
            action_prob = probs[4]
            
            # Get severity prediction
            severity_idx = severity_probs.argmax()
            severity_label = self.severity_labels[severity_idx]
            severity_confidence = float(severity_probs[severity_idx])
            
            # Get action prediction
            action_label = self.action_labels[1] if action_prob > 0.5 else self.action_labels[0]
            action_confidence = float(action_prob if action_prob > 0.5 else 1 - action_prob)
            
            return {
                "severity": {
                    "label": severity_label,
                    "confidence": f"{severity_confidence:.2%}"
                },
                "action": {
                    "label": action_label,
                    "confidence": f"{action_confidence:.2%}"
                }
            }

def process_reddit_post(classifier: RedditPostClassifier, post_data: Dict) -> Dict:
    text = post_data["raw_data"]["text"]
    predictions = classifier.predict(text)
    
    return {
        "predictions": {
            "severity": predictions["severity"],
            "action": predictions["action"]
        }
    }

if __name__ == "__main__":
    # Initialize classifier
    classifier = RedditPostClassifier("multilabel_model/checkpoint-10")
    
    # Load labels (though now we know the structure)
    classifier.load_labels_from_dataset("dataset/medical_mimic(1).json")
    classifier.initialize_model()
    
    # Load test data
    with open("dataset/medical_mimic(1).json", 'r') as f:
        test_data = json.load(f)[0]
    
    # Process and print results
    result = process_reddit_post(classifier, test_data)
    print(json.dumps(result, indent=2))