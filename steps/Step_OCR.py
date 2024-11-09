from base import VLMStep
from step_decorator import Step

from typing import Union, Dict

import yaml

import re

from client import client

from utils import get_config_file

@Step("ocr")
class Step_OCR(VLMStep):
    
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
        # Remove leading and trailing whitespaces
        text = output.strip()

        # Pattern for question: Match everything between "**Question:**" and "**Answer:**"
        question_pattern = r'\*\*Question:\*\*\s*(.*?)\s*\*\*Answer:'

        # Pattern for answer: Match everything after "**Answer:**"
        answer_pattern = r'\*\*Answer:\*\*\s*(.*)$'

        # Extract question
        question_match = re.search(question_pattern, text, re.DOTALL)
        if not question_match:
            raise ValueError("Could not find Question Section")
        question = question_match.group(1).strip()

        # Extract answer
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if not answer_match:
            raise ValueError("Could not find Answer Section")
        answer = answer_match.group(1).strip()

        return {
            "question": question,
            "answer": answer,
            "output": output
        }
