import json
import argparse
from typing import Any
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime


class FillInCSV:
    def __init__(self, json_file, model):
        self.json_file = json_file
        self.model = model

    def output_state(self) -> dict:
        try:
            with open(self.json_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: JSON file '{self.json_file}' not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return {}

    def get_state(self, key: str) -> Any:
        state = self.output_state()
        try:
            return state.get(key)
        except KeyError:
            print(f"Error: Key '{key}' not found in state.")
            return None

    def process_image(self, image, csv, dtype):
        try:
            result = self.get_state(image)
            if result is None:
                return

            row_index = csv[csv["annotation_url"].str.contains(image)].index

            if len(row_index) == 0:
                print(f"Error: No row found with annotation_url '{image}'.")
                return
            elif len(row_index) > 1:
                print(f"Error: Multiple rows found with annotation_url '{image}'.")
                return

            updated_row = csv.loc[row_index[0]].copy()

            if "1.1.1" in result:
                updated_row["pred_err_1.1.1"] = result["1.1.1"]
            if "1.1.2" in result:
                updated_row["err_reasoning_1.1.2"] = result["1.1.2"]["reasoning"]
                updated_row["pred_err_1.1.2"] = result["1.1.2"]["error"]

            if "ocr" in result:
                updated_row["pred_ocr_q_1.2.1_1.2.2_2.2_3.2"] = result["ocr"]["question"]
                updated_row["pred_ocr_a_1.2.1_1.2.2_2.2_3.2"] = result["ocr"]["answer"]

            if "1.2.1" in result:
                updated_row["pred_err_1.2.1"] = result["1.2.1"]

            if "1.2.2" in result:
                updated_row["pred_err_1.2.2"] = result["1.2.2"]["error"]
                updated_row["err_reasoning_1.2.2"] = result["1.2.2"]["reasoning"]

            if "2.1" in result:
                updated_row["pred_reasoning_2.1"] = result["2.1"]["reasoning"]
                updated_row["pred_eloc_2.1"] = result["2.1"]["error_localization"]

            if "2.2" in result:
                updated_row["pred_reasoning_2.2"] = result["2.2"]["reasoning"]
                updated_row["pred_eloc_2.2"] = result["2.2"]["error_localization"]

            if "3.1" in result:
                updated_row["pred_reasoning_3.1"] = result["3.1"]["reasoning"]
                updated_row["pred_corr_a_3.1"] = result["3.1"]["corrected_answer_latex"]

            if "3.2" in result:
                updated_row["pred_reasoning_3.2"] = result["3.2"]["reasoning"]
                updated_row["pred_corr_a_3.2"] = result["3.2"]["corrected_answer_latex"]

            return updated_row
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            return None

    def execute(self, csv_file) -> None:
        try:
            print("Outputting the state")

            images = self.get_state("images")
            if images is None:
                print("No images found in state.")
                return

            try:
                csv = pd.read_csv(csv_file, dtype={"annotation_url": str})
            except FileNotFoundError:
                print(f"Error: CSV file '{csv_file}' not found.")
                return
            except pd.errors.ParserError as e:
                print(f"Error reading CSV file: {e}")
                return

            updated_rows = []
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self.process_image, image, csv, None) for image in images]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                    updated_row = future.result()
                    if updated_row is not None:
                        updated_rows.append(updated_row)

            if updated_rows:
                updated_csv = pd.DataFrame(updated_rows)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                updated_csv.to_csv(f"{self.model}_{timestamp}.csv", index=False)
                print("State outputted successfully")
            else:
                print("No rows were updated.")

        except Exception as e:
            print(f"Error executing the process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill in CSV with JSON data")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--model', type=str, required=True, help='Model name')

    args = parser.parse_args()

    fill_in_csv = FillInCSV(args.json_file, args.model)
    fill_in_csv.execute(args.csv_file)