from base import VLMStep
from step_decorator import Step

from typing import Union, Dict

import yaml
import base64

import re

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

@Step("1.1.2")
class Step_1_1_2(VLMStep):
    
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
    
    def parse_output(self, output: str) -> Dict|None:
        # Remove Leading and Trailing Whistespaces

        text = output.strip()

        # Pattern for reasoning: Match everything between "**Reasoning:**" and "**Error:**"
        reasoning_pattern = r'\*\*Reasoning:\*\*\s*(.*?)\s*\*\*Error:'
        
        # Pattern for error: Match the number (0 or 1) after "**Error:**"
        error_pattern = r'\*\*Error:\*\*\s*(\d+)'
        
        # Extract reasoning
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        if not reasoning_match:
            raise ValueError("Could not find reasoning section")
        reasoning = reasoning_match.group(1).strip()
        
        # Extract error value
        error_match = re.search(error_pattern, text)
        if not error_match:
            raise ValueError("Could not find error value")
        error = int(error_match.group(1))
        
        return {
            "reasoning": reasoning,
            "error": error
        }
