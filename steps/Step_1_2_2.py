from base import VLMStep
from step_decorator import Step

from typing import Union, Dict

import yaml
import base64

import re

from client import client

from utils import get_config_file

@Step("1.2.2")
class Step_1_2_2(VLMStep):
    
    max_tokens = 1024
    temperature = 0

    def __init__(self, model_path: str):
        self.experiment_id = self.get_id()
        self.model_path = model_path

        config_file = get_config_file()

        # Load the config.yaml file
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)
            self.user_prompt = self.config["experiments"][self.experiment_id]["user_prompt"]
            self.system_prompt = self.config["experiments"][self.experiment_id]["system_prompt"]

    def hit(self, image_url):
        image = self.encode_image(image_url)


        ## Add Question and Answer in the prompt
        try:
            q, a = self.get_ocr(image_url)

            processed_prompt = f'{self.user_prompt}\n\n**Question:** {q}\n\n**Answer:** {a}'

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
                        {"type": "text", "text": processed_prompt},
                        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    ],
                }]
            )
            return(chat_response.choices[0].message.content)
        
        except Exception as e:
            print(e)
            return None
    
    def parse_output(self, output: str) -> Union[Dict, None]:
        # Remove leading and trailing whitespaces
        text = output.strip()

        # Pattern for reasoning: Match everything between "**Reasoning:**" and "**Error:**"
        reasoning_pattern = r'\*\*Reasoning:\*\*\s*(.*?)\s*\*\*Error:'

        # Pattern for error: Match the number 0 or 1 after "**Error:**" with validation for single digit only
        error_pattern = r'\*\*Error:\*\*\s*([01])'

        # Extract reasoning
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        if not reasoning_match:
            raise ValueError("Could not find reasoning section")
        reasoning = reasoning_match.group(1).strip()

        # Extract error value, ensuring it's a single binary digit (0 or 1)
        error_match = re.search(error_pattern, text)
        if not error_match:
            raise ValueError("Could not find or validate error value as binary (0 or 1)")
        error = int(error_match.group(1))

        return {
            "reasoning": reasoning,
            "error": error,
            "output": output
        }
