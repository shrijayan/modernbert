# main.py
import os
from config import ModelConfig, ClassificationType
from pipeline import Pipeline
from model_push import push_trained_model

def main():
    # Initialize and run the training pipeline
    config = ModelConfig()
    pipeline = Pipeline(config)
    pipeline.run()  # Uses default dataset "shrijayan/medical_mimic"
    
    # Push the trained model to HuggingFace Hub
    model_path = "./multilabel_model_final"  # Path where the model was saved
    
    # Get HuggingFace token from environment variable
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")
    
    # Push the model to HuggingFace Hub
    push_trained_model(
        model_path=model_path,
        repo_name="medical-severity-classifier",
        token=hf_token,
        organization=None  # Set this to your organization name if needed
    )

if __name__ == "__main__":
    main()