from typing import List, Union, Optional, Dict, Any
from base import BaseStep
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



class VLMStep(BaseStep):

    experiment_id = None

    def hit(self, image_url: str) -> Any:
            raise NotImplementedError("Subclasses must implement hit method")
    
    def parse_output(self, image_url: str) -> Union[int, Dict, None]:
        raise NotImplementedError("Subclasses must implement parse_output method")
    
    def execute(self):
        
        # Get images

        images = self.get_state("images")

        # Process all the images parallely and take advantage of internal batching of vLLM

        results = []
        total_images = len(images)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_image = {executor.submit(self.hit, img): img for img in images}
    
            with tqdm(total=total_images, desc="Processing Images") as pbar:
                
                for future in as_completed(future_to_image):
                    # Append both image name and the result to the results array
                    output = future.result()
                    img_name = future_to_image[future]
                    results.append((img_name, output))
                    pbar.update(1)
        
        # Update State of each of the images accordingly

        for result in results:
        
            img_name, output = result
        
            self.update_state(image = img_name, result = output)
    

    def encode_image(self, image: str) -> str:
        with open(f"../images/{image}", 'rb') as img_file:
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

        return
            

    