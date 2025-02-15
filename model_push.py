# model_push.py
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import Optional

class ModelPusher:
    def __init__(self, model_path: str, repo_name: str, organization: Optional[str] = None):
        """
        Initialize ModelPusher with model path and target repository details.
        
        Args:
            model_path: Local path to the saved model
            repo_name: Name for the HuggingFace repository
            organization: Optional organization name to push under
        """
        self.model_path = model_path
        self.repo_name = repo_name
        self.organization = organization
        self.api = HfApi()
        
    def create_model_card(self) -> str:
        """Create a basic model card with relevant information."""
        return """---
language: en
tags:
- medical-text-classification
- multilabel-classification
datasets:
- shrijayan/medical_mimic
model-index:
- name: ModernBERT Medical Severity Classifier
  results:
  - task: 
      name: Medical Text Classification
      type: text-classification
---

# ModernBERT Medical Severity Classifier

This model is fine-tuned on medical text data to classify severity levels (low, medium, high, critical) and determine if immediate action is required.

## Model Description

- **Model Type:** ModernBERT-base fine-tuned for multi-label classification
- **Task:** Medical text severity classification and action requirement detection
- **Training Data:** MIMIC medical dataset
- **Labels:** 
  - Severity: low, medium, high, critical
  - Action: immediate action required (yes/no)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/medical-severity-classifier")
model = AutoModelForSequenceClassification.from_pretrained("YOUR_USERNAME/medical-severity-classifier")

# Prepare input
text = "Patient presents with severe chest pain and shortness of breath"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
```

## Limitations and Bias

This model should be used as a tool to assist medical professionals, not as a replacement for professional medical judgment. The model's predictions should always be verified by qualified healthcare providers.
"""

    def push_to_hub(self, token: str, commit_message: str = "Initial model upload") -> None:
        """
        Push the model, tokenizer, and model card to HuggingFace Hub.
        
        Args:
            token: HuggingFace API token
            commit_message: Commit message for the upload
        """
        # Set the token
        os.environ['HUGGINGFACE_TOKEN'] = token
        
        # Create the full repository name
        repo_id = f"{self.organization}/{self.repo_name}" if self.organization else f"{self.repo_name}"
        
        try:
            # Create repository if it doesn't exist
            self.api.create_repo(
                repo_id=repo_id,
                private=False,
                exist_ok=True
            )
            
            # Load the model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Push the model and tokenizer
            model.push_to_hub(repo_id, use_auth_token=token)
            tokenizer.push_to_hub(repo_id, use_auth_token=token)
            
            # Create and push model card
            model_card = self.create_model_card()
            self.api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            print(f"Successfully pushed model to {repo_id}")
            
        except Exception as e:
            print(f"Error pushing model to hub: {str(e)}")
            raise

def push_trained_model(
    model_path: str,
    repo_name: str,
    token: str,
    organization: Optional[str] = None
) -> None:
    """
    Convenience function to push a trained model to HuggingFace Hub.
    
    Args:
        model_path: Path to the saved model
        repo_name: Name for the HuggingFace repository
        token: HuggingFace API token
        organization: Optional organization name to push under
    """
    pusher = ModelPusher(model_path, repo_name, organization)
    pusher.push_to_hub(token)