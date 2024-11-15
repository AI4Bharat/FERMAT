from steps import Step_OCR
from typing import Union, Dict
from step_decorator import Step

class Pixtral_Step_OCR(Step_OCR):
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
        answer_start = text.find("**Answer:**", question_start)
        if answer_start == -1:
            answer_start = text.find("\\textbf{Answer:}", question_start)
        if answer_start == -1:
            raise ValueError("Could not find Answer Section")
        
        question = text[question_start:answer_start].strip()
        
        answer_start += len("**Answer:**") if "**Answer:**" in text else len("\\textbf{Answer:}")
        
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