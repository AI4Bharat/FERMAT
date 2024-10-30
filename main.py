from typing import List
from base import BaseStep
import pandas as pd

from utils import get_model_path, get_csv

class Pipeline:
    
    def __init__(self):
        self.steps: List[BaseStep] = []

    def add_step(self, step: BaseStep) -> None:
        self.steps.append(step)

    def get_step(self, step_id: str = None) -> BaseStep:
        
        if step_id is None:
            return self.steps[0]
        
        for step in self.steps:
            if step.get_id() == step_id:
                return step

    def run(self) -> None:
        for step in self.steps:
            print(f"Executing step: {step.get_id()}")
            step.execute()
            print(f"Completed step: {step.get_id()}")


def load_images(url: str) -> List[str]:
    csv = pd.read_csv(url)

    img_names = csv[:1]["annotation_url"]

    return list(map(lambda x: x.split("/")[-1], img_names))

from steps import Llama_Step_1_1_1, Llama_Step_1_1_2, Step_1_1_1, Step_1_1_2, OutputState

def get_steps(model: str):
    models = {
        "llama": [Llama_Step_1_1_1, Llama_Step_1_1_2],
        "default": [Step_1_1_1, Step_1_1_2]
    }
    
    return models.get(model, models["default"])



if __name__ == "__main__":

    model = "llama"

    model_path = get_model_path(model)
    url = get_csv()

    pipeline = Pipeline()
    
    steps = get_steps(model)

    for step in steps:
        pipeline.add_step(step(model_path = model_path))

    pipeline.add_step(OutputState(model=model))

    pipeline.get_step().set_state("images", load_images(url))
    pipeline.get_step().set_state("not_parsed", [])
    
    pipeline.run()