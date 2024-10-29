import yaml

def get_csv():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config['csv']