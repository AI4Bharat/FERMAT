from typing import List, Union, Optional, Dict, Any
from base import BaseStep
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ImageNotFoundException(Exception):
    pass

class OCRNotFoundException(Exception):
    pass

class VLMStep(BaseStep):

    experiment_id = None

    def hit(self, image_url: str) -> Any:
            raise NotImplementedError("Subclasses must implement hit method")
    
    def parse_output(self, output: str) -> Union[Dict, None]:
        raise NotImplementedError("Subclasses must implement parse_output method")
    
    def execute(self):
        
        # Get images
        images = self.get_state("images")

        # Process all the images in parallel with vLLM batching
        results = []
        total_images = len(images)

        with tqdm(total=total_images, desc="Processing Images") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.hit, images[i]): images[i] for i in range(total_images)}

                for future in as_completed(futures):
                    img_name = futures[future]
                    try:
                        output = future.result()
                        if output is not None:
                            results.append((img_name, output))
                        else:
                            print(f"Error processing image {img_name}: No output returned by VLM")
                            not_parsed = self.get_state("not_parsed")

                            not_parsed.append({
                                "image": img_name,
                                "result": output,
                                "experiment_id": self.experiment_id
                            })

                            self.set_state("not_parsed", not_parsed)
                            
                    except Exception as e:
                        print(f"Error processing image {img_name}: {e}")
                    finally: 
                        pbar.update(1)

        # Update state for each processed image
        for img_name, output in results:
            self.update_state(image=img_name, result=output)

    def encode_image(self, image: str) -> str:
        with open(f"../pqa_images_grayscale/{image}", 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
        
    def update_state(self, image: str, result: Union[str]) -> None:
        """
        Update the state of the image with the results
        """

        try:
            current_state = self.get_state(image)

            if current_state is None:
                current_state = {}
            
            parsed_output = self.parse_output(result)

            if parsed_output:
                
                current_state[self.experiment_id] = parsed_output
                
                self.set_state(image, current_state)

            else:

                not_parsed = self.get_state("not_parsed")

                not_parsed.append({
                    "image": image,
                    "result": result,
                    "experiment_id": self.experiment_id
                })

                self.set_state("not_parsed", not_parsed)
        
        except Exception as e:
            
            print(f"Error while updating the state of the image: {e}")

            not_parsed = self.get_state("not_parsed")

            not_parsed.append({
                "image": image,
                "result": result,
                "experiment_id": self.experiment_id
            })

            self.set_state("not_parsed", not_parsed)

        return
            



    def get_ocr(self, image: str) -> List[str]:
        """
        Get the OCR of the image
        """
        try:
            result = self.get_state(image)
            
            if result is None:
                raise ImageNotFoundException(f"No object for image {image} found in the state")
            
            ocr = result.get("ocr", None)
            
            if ocr is not None:
                question = ocr.get("question", None)
                answer = ocr.get("answer", None)

                if question is None or answer is None:
                    raise OCRNotFoundException("OCR object does not contain question or answer")

                return [question, answer]
            else:
                raise OCRNotFoundException("OCR not found in the result")
        except Exception as e:
            raise e