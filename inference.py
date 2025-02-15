# inference.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class MedicalBERTInference:
    def __init__(self, model_path: str = "./multilabel_model_final"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.severity_mapping = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        # Process outputs
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Severity prediction (multi-class)
        severity_probs = probs[:4]
        severity_idx = np.argmax(severity_probs)
        severity = self.severity_mapping[severity_idx]
        
        # Action required prediction (binary)
        action_required = bool(probs[4] > 0.5)
        
        return {
            "severity": severity,
            "action_required": action_required,
            "severity_probabilities": {
                self.severity_mapping[i]: float(severity_probs[i]) 
                for i in range(4)
            },
            "action_probability": float(probs[4])
        }

# Example usage
if __name__ == "__main__":
    inference = MedicalBERTInference()
    sample_text = "Individual purchased firearm with suicidal intent, has plan, note, and videos prepared"
    result = inference.predict(sample_text)
    print("Prediction Result:")
    print(f"Severity: {result['severity']}")
    print(f"Action Required: {result['action_required']}")
    print("Severity Probabilities:", result['severity_probabilities'])
    print(f"Action Probability: {result['action_probability']:.4f}")