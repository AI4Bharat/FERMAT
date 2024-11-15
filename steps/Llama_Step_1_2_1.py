from steps import Step_1_2_1
from typing import Any


from client import client

class Llama_Step_1_2_1(Step_1_2_1):

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
                    "role": "user",
                    "content": [
                        {"type": "text", "text": processed_prompt},
                        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    ],
                }]
            )
            return(chat_response.choices[0].message.content)
        
        except Exception as e:

            print(e, image_url)
            return None