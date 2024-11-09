from base import BaseStep
from step_decorator import Step
import pandas as pd
from utils import get_csv
import json

@Step("OutputState")
class OutputState(BaseStep):
    def __init__(self, model: str):
        self.model = model

    def execute(self) -> None:
        print("Outputting the state")

        images = self.get_state("images")
        not_parsed = self.get_state("not_parsed")

        # Dump the contents of the state object to a JSON file

        with open(f"state_{self.model}.json", "w") as file:
            json.dump(self.output_state(), file, indent=4)

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
            "pred_ocr_q_1.2.1_1.2.2_2.2_3.2": str,
            "pred_ocr_a_1.2.1_1.2.2_2.2_3.2": str,
            "pred_err_1.2.1": float,
            "pred_err_1.2.2": float,
            "err_reasoning_1.2.2": str,
            "pred_reasoning_2.1": str,
            "pred_eloc_2.1": str,
            "pred_reasoning_2.2": str,
            "pred_eloc_2.2": str,
            "pred_reasoning_3.1": str,
            "pred_corr_a_3.1": str,
            "pred_reasoning_3.2": str,
            "pred_corr_a_3.2": str
        }
        csv = pd.read_csv(url, dtype=dtype)

        # Loop through each image.
        for image in images:
            # Get the corresponding result
            result = self.get_state(image)

            if result is None:
                continue

            # Get the row_index
            row_index = csv[csv["annotation_url"].str.contains(image)].index[0]

            # Update the CSV
            csv.loc[row_index, "model_name"] = self.model

            if "1.1.1" in result:
                csv.loc[row_index, "pred_err_1.1.1"] = (result["1.1.1"])
            if "1.1.2" in result:
                csv.loc[row_index, "err_reasoning_1.1.2"] = result["1.1.2"]["reasoning"]
                csv.loc[row_index, "pred_err_1.1.2"] = (result["1.1.2"]["error"])

            if "ocr" in result:
                csv.loc[row_index, "pred_ocr_q_1.2.1_1.2.2_2.2_3.2"] = result["ocr"]["question"]
                csv.loc[row_index, "pred_ocr_a_1.2.1_1.2.2_2.2_3.2"] = result["ocr"]["answer"]
            
            if "1.2.1" in result:
                csv.loc[row_index, "pred_err_1.2.1"] = (result["1.2.1"])

            if "1.2.2" in result:
                csv.loc[row_index, "pred_err_1.2.2"] = (result["1.2.2"]["error"])
                csv.loc[row_index, "err_reasoning_1.2.2"] = result["1.2.2"]["reasoning"]

            if "2.1" in result:
                csv.loc[row_index, "pred_reasoning_2.1"] = result["2.1"]["reasoning"]
                csv.loc[row_index, "pred_eloc_2.1"] = result["2.1"]["error_localization"]

            if "2.2" in result:
                csv.loc[row_index, "pred_reasoning_2.2"] = result["2.2"]["reasoning"]
                csv.loc[row_index, "pred_eloc_2.2"] = result["2.2"]["error_localization"]
            
            if "3.1" in result:
                csv.loc[row_index, "pred_reasoning_3.1"] = result["3.1"]["reasoning"]
                csv.loc[row_index, "pred_corr_a_3.1"] = result["3.1"]["corrected_answer_latex"]

            if "3.2" in result:
                csv.loc[row_index, "pred_reasoning_3.2"] = result["3.2"]["reasoning"]
                csv.loc[row_index, "pred_corr_a_3.2"] = result["3.2"]["corrected_answer_latex"]

        # Save the updated CSV back to the file
        csv.to_csv(f"{self.model}.csv", index=False)

            