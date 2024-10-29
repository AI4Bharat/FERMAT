import yaml

def get_model_path(model: str) -> str:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config['models'][model]