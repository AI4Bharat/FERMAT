from steps import Step_2_2
from typing import Any
from client import client
from utils import replace_unicode

class Pixtral_Step_2_2(Step_2_2):

    def hit(self, image_url):
        image = self.encode_image(image_url)

        ## Add Question and Answer in the prompt
        
        try:

            q, a = self.get_ocr(image_url)

            processed_prompt = f'{self.user_prompt}\n\n**Question:** {q}\n\n**Answer:** {a}'

            processed_prompt = replace_unicode(processed_prompt)


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

            print(e, image_url)
            return None