from steps import Step_1_1_2
from typing import Any, Union, Dict


from client import client

class Llama_Step_1_1_2(Step_1_1_2):

    def hit(self, image_url: str) -> Any:
        image = self.encode_image(image_url)


        chat_response = client.chat.completions.create(
            model=self.model_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
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
        
        # Find the Reasoning section
        reasoning_start = text.find("**Reasoning:**")
        if reasoning_start == -1:
            reasoning_start = text.find("\\textbf{Reasoning:}")
        if reasoning_start == -1:
            raise ValueError("Could not find Reasoning Section")
        
        reasoning_start += len("**Reasoning:**") if "**Reasoning:**" in text else len("\\textbf{Reasoning:}")
        
        # Find the Error and Answer sections
        error_start = text.find("**Error:**", reasoning_start)
        answer_start = text.find("**Answer:**", reasoning_start)
        
        if error_start == -1 and answer_start == -1:
            raise ValueError("Could not find Error or Answer Section")
        
        # Select the one that occurs later
        if error_start != -1 and answer_start != -1:
            section_start = max(error_start, answer_start)
            section_type = "error" if section_start == error_start else "answer"
        else:
            # Only one of them is present
            section_start = answer_start if answer_start != -1 else error_start
            section_type = "answer" if answer_start != -1 else "error"
        
        reasoning = text[reasoning_start:section_start].strip()
        
        section_start += len("**Error:**") if section_type == "error" else len("**Answer:**")
        section_end = section_start + 2  # We only need the next character which should be a digit
        
        digit = text[section_start:section_end].strip()

        return {
            "reasoning": reasoning,
            "error": digit,
            "output": output
        }