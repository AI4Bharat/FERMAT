from base import VLMStep
from step_decorator import Step

from typing import Union, Dict

import yaml

import re

from client import client

from utils import get_config_file

@Step("2.1")
class Step_2_1(VLMStep):
    
    max_tokens = 1024
    temperature = 0

    def __init__(self, model_path: str):
        self.experiment_id = self.get_id()
        self.model_path = model_path

        config_file = get_config_file()

        # Load the config.yaml file
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)
            print(self.config["experiments"].keys())
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
    

    def parse_output(self, output: str) -> Union[Dict, None]:
        text = output.strip()

        # Pattern for reasoning
        reasoning_pattern = r'\*\*Reasoning:\*\*\s*(.*?)\s*\*\*Error Localization:'
        
        # Pattern for error localization
        error_localization_pattern = r'\*\*Error Localization:\*\*\s*(.*)$'  # Match until end of string

        # Extract reasoning
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        if not reasoning_match:
            raise ValueError("Could not find Reasoning section")
        reasoning = reasoning_match.group(1).strip()

        # Extract error localization
        error_localization_match = re.search(error_localization_pattern, text, re.DOTALL)
        if not error_localization_match:
            raise ValueError("Could not find Error Localization section")
        error_localization = error_localization_match.group(1).strip()

        return {
            "reasoning": reasoning,
            "error_localization": error_localization,
            "output": output
        }

