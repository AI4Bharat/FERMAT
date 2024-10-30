from base import VLMStep
from step_decorator import Step

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from typing import Union, Dict

import yaml
import re

from client import client

@Step("1.1.1")
class Step_1_1_1(VLMStep):

    max_tokens = 2048
    temperature = 0

    def __init__(self, model_path: str):

        self.experiment_id = self.get_id()
        
        self.model_path = model_path

        # Load the config.yaml file
        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
            self.user_prompt = self.config["experiments"][self.experiment_id]["user_prompt"]
            self.system_prompt = self.config["experiments"][self.experiment_id]["system_prompt"]

    
    def hit(self, image_url):
        image = self.encode_image(image_url)


        chat_response = client.chat.completions.create(
            model=self.model_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                    ],
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ],
            }]
        )
        return(chat_response.choices[0].message.content)
        
    def parse_output(self, output: str) -> Union[int, None]:
        # Remove Leading and Trailing Whistespaces
        
        output = output.strip()

        match = re.search(r'\d+', output)

        if match:
            output = match.group(0)
        else:
            raise ValueError("No number found in the output")
        
        return output

