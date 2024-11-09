from base import VLMStep
from step_decorator import Step

from typing import Union, Dict

import yaml

import re

from client import client

from utils import get_config_file

@Step("3.2")
class Step_3_2(VLMStep):
    
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
        text = output.strip()

        # Pattern for reasoning
        reasoning_pattern = r'\*\*Reasoning:\*\*\s*(.*?)\s*\*\*Corrected Answer LaTeX:'
        
        # Pattern for corrected answer LaTeX
        corrected_answer_latex_pattern = r'\*\*Corrected Answer LaTeX:\*\*\s*(.*)$'  # Capture until end of string

        # Extract reasoning
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        if not reasoning_match:
            raise ValueError("Could not find Reasoning section")
        reasoning = reasoning_match.group(1).strip()

        # Extract corrected answer in LaTeX
        corrected_answer_latex_match = re.search(corrected_answer_latex_pattern, text, re.DOTALL)
        if not corrected_answer_latex_match:
            raise ValueError("Could not find Corrected Answer LaTeX section")
        corrected_answer_latex = corrected_answer_latex_match.group(1).strip()

        return {
            "reasoning": reasoning,
            "corrected_answer_latex": corrected_answer_latex,
            "output": output
        }
