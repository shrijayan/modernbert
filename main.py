# main.py
from config import ModelConfig, ClassificationType
from pipeline import Pipeline

def main():
    config = ModelConfig()
    pipeline = Pipeline(config)
    pipeline.run()  # Uses default dataset "shrijayan/medical_mimic"

if __name__ == "__main__":
    main()