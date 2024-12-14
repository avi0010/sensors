import os
from tqdm.contrib.concurrent import process_map
import pandas as pd

SHEETS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "BME"]
ENVIRONMENT = ["Temperature Deriv.", "Humidity Deriv."]
OUTPUTS = ["Exposure"]

INPUTS = SHEETS + ENVIRONMENT
SENSORS = INPUTS + OUTPUTS

PARAMETER = "feature"
WARMUP = 800

SAVE_DIR = "data_generated"

def create_dataset(filename: str):
    try:
        # Initialize the DataFrame with exposure, temperature, and humidity columns
        df = pd.DataFrame()

        # Load and process the S0 sheet separately for initial exposure, temperature, and humidity
        S0_db = pd.read_excel(filename, sheet_name="S0")
        S0_db = S0_db.infer_objects()
        S0_filtered = S0_db[S0_db["Routine Counter"] >= WARMUP]
        df = S0_db[ENVIRONMENT + OUTPUTS].join(df)

        for sheet_name in SHEETS:  # Adjust the range if the number of sheets is different
            sheet_df = pd.read_excel(filename, sheet_name=sheet_name)
            # Extract the relevant column and assign it to the main DataFrame

            sheet_df = sheet_df.infer_objects()

            #if sheet_name != "BME":
            #    sheet_df[PARAMETER] = 20000 * sheet_df["Raw Deriv."] - sheet_df["Temperature Deriv."] - sheet_df["Humidity Deriv."]
            #else:
            sheet_df[PARAMETER] = sheet_df["Raw Deriv."]

            filtered_df = sheet_df[sheet_df["Routine Counter"] >= WARMUP]
            # Extract the relevant column and assign it to the main DataFrame
            df[sheet_name] = sheet_df[PARAMETER]
    
        df.replace(-999, 0, inplace=True)

        df.to_excel(os.path.join(SAVE_DIR, filename.split("/")[-1]), index=False) 
    except Exception as e:
        print(f"{filename}: {e}")
    return


if __name__ == '__main__':
    os.mkdir(SAVE_DIR)

    # Train
    input_files = []
    training_dir = "data/train/"
    for file in os.listdir(training_dir):
        input_files.append(os.path.join(training_dir, file))


    val_dir = "data/val/"
    for file in os.listdir(val_dir):
        input_files.append(os.path.join(val_dir, file))

    r = process_map(create_dataset, input_files)
