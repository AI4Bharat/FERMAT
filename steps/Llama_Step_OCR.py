from steps import Step_OCR
from typing import Any, Union, Dict


from client import client

class Llama_Step_OCR(Step_OCR):

    def hit(self, image_url):
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
        
        # Handle both latex code blocks and markdown format
        if text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        
        # Find the question section
        question_start = text.find("**Question:**")
        if question_start == -1:
            question_start = text.find("\\textbf{Question:}")
        if question_start == -1:
            raise ValueError("Could not find Question Section")
        
        question_start += len("**Question:**") if "**Question:**" in text else len("\\textbf{Question:}")
        
        # Find the answer section
        answer_start = text.find("\\text{Answer:}", question_start)
        if answer_start == -1:
            answer_start = text.find("Answer:", question_start)
        if answer_start == -1:
            raise ValueError("Could not find Answer Section")
        
        question = text[question_start:answer_start].strip()
        
        answer_start += len("\\text{Answer:}") if "Answer:" in text else len("Answer:")
        
        # Find the end of the answer section
        answer_end = text.find("```", answer_start)
        if answer_end == -1:
            answer_end = len(text)
        
        answer = text[answer_start:answer_end].strip()

        # Clean up any remaining latex/markdown artifacts
        question = question.replace('\\n', '\n')
        answer = answer.replace('\\n', '\n')

        return {
            "question": question,
            "answer": answer,
            "output": output
        }