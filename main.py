# main.py
from config import ModelConfig, ClassificationType
from pipeline import Pipeline

def main():
    config = ModelConfig()
    pipeline = Pipeline(config)
    pipeline.run('dataset/medical_mimic(1).json', 'dataset/medical_mimic_new_(1).json')

if __name__ == "__main__":
    main()