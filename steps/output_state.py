from base import BaseStep
from step_decorator import Step
import pandas as pd
from utils import get_csv

@Step("OutputState")
class OutputState(BaseStep):
    def __init__(self, model: str):
        self.model = model

    def execute(self) -> None:
        # Convert to CSV afterwards and Output the file
        print("Outputting the state")

        images = self.get_state("images")
        not_parsed = self.get_state("not_parsed")

        # Dump the contents of not_parsed to a text file
        with open(f"not_parsed_{self.model}.txt", "w") as file:
            for item in not_parsed:
                item = f"{item['image']};{item['result']};{item['experiment_id']}"
                file.write(f"{item}\n")

        # Get the CSV file
        url = get_csv()

        # Read the CSV file with specified dtype to avoid warnings
        dtype = {
            "annotation_url": str,
            "model_name": str,
            "pred_err_1.1.1": float,
            "err_reasoning_1.1.2": str,
            "pred_err_1.1.2": float
        }
        csv = pd.read_csv(url, dtype=dtype)

        # Loop through each image.
        for image in images:
            # Get the corresponding result
            result = self.get_state(image)

            # Get the row_index
            row_index = csv[csv["annotation_url"].str.contains(image)].index[0]

            # Update the CSV
            csv.loc[row_index, "model_name"] = self.model

            if "1.1.1" in result:
                csv.loc[row_index, "pred_err_1.1.1"] = float(result["1.1.1"])
            if "1.1.2" in result:
                if "reasoning" in result["1.1.2"]:
                    csv.loc[row_index, "err_reasoning_1.1.2"] = result["1.1.2"]["reasoning"]
                if "error" in result["1.1.2"]:
                    csv.loc[row_index, "pred_err_1.1.2"] = float(result["1.1.2"]["error"])

        # Save the updated CSV back to the file
        csv.to_csv(f"{self.model}.csv", index=False)

            