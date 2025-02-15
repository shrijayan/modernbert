# main.py
import os
from datetime import datetime
from config import ModelConfig
from pipeline import Pipeline

def main():
    # Initialize configuration
    config = ModelConfig()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./models/medical_classifier_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = Pipeline(config)
    pipeline.run(config.dataset_name)
    
if __name__ == "__main__":
    main()
