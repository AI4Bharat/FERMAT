from typing import List
from base import BaseStep
import pandas as pd
import json
from utils import get_model_path, get_csv

model = "llama"

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
            with open(f"state_{model}_pqa_{step.get_id()}.json", "w") as file:
                json.dump(step.output_state(), file, indent=4)
            

def load_images(url: str) -> List[str]:
    image_urls = []
    import os
    dir_name = '../pqa_images_grayscale'
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.png'):
                image_urls.append(file)

    return image_urls

# def load_images(url: str) -> List[str]:
#     csv = pd.read_csv(url)

#     img_names = csv["annotation_url"]

#     return list(map(lambda x: x.split("/")[-1], img_names))

from steps import Pixtral_Step_OCR, Llama_Step_1_1_1, Llama_Step_1_1_2, Step_1_1_1, Step_1_1_2, OutputState, Step_1_2_1, Step_1_2_2, Step_2_1, Step_2_2, Step_3_1, Step_3_2, Step_OCR, Pixtral_Step_1_2_1, Pixtral_Step_1_2_2, Pixtral_Step_2_2, Pixtral_Step_3_2, Llama_Step_1_2_2, Llama_Step_1_2_1, Llama_Step_2_1, Llama_Step_2_2, Llama_Step_3_1, Llama_Step_3_2, Llama_Step_OCR
import json

def get_steps(model: str):
    models = {
        "llama": [Llama_Step_1_1_1, Llama_Step_1_1_2, Llama_Step_OCR, Llama_Step_1_2_1, Llama_Step_1_2_2, Llama_Step_2_1, Llama_Step_2_2, Llama_Step_3_1, Llama_Step_3_2],
        "pixtral": [Step_1_1_1, Step_1_1_2, Pixtral_Step_OCR, Pixtral_Step_1_2_1, Pixtral_Step_1_2_2, Step_2_1, Pixtral_Step_2_2, Step_3_1, Pixtral_Step_3_2],
        "default": [Step_1_1_1, Step_1_1_2, Step_OCR, Step_1_2_1, Step_1_2_2, Step_2_1, Step_2_2, Step_3_1, Step_3_2]
    }
    
    return models.get(model, models["default"])



if __name__ == "__main__":
    model_path = get_model_path(model)
    url = get_csv()

    pipeline = Pipeline()
    
    steps = get_steps(model)

    for step in steps:
        pipeline.add_step(step(model_path = model_path))

    # pipeline.add_step(OutputState(model=model))

    pipeline.get_step().set_state("images", load_images(url))
    pipeline.get_step().set_state("not_parsed", [])

    # with open('../llama.json') as f:
    #     data = json.load(f)
    #     data["images"] = data["images"][:5]
    #     pipeline.get_step().put_state(data)

    pipeline.run()
