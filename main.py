# main.py
from config import ModelConfig
from pipeline import Pipeline

def main():
    config = ModelConfig()

    pipeline = Pipeline(config)
    pipeline.run(config.dataset_name)
    
if __name__ == "__main__":
    main()
