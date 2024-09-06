import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
# import plotly.express as px
import os
from tqdm.contrib.concurrent import process_map
import joblib
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
SENSORS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "Temperature", "Humidity", ]
PARAMETER = "Derivative"
WARMUP = 600
POSITIVE_LABEL_TIME = 256
MM_FILE_NAME = f"scaler_{PARAMETER}.gz"
LENGTH = 100

class SequenceDataset(Dataset):
    def __init__(self, dir:str):
        self.files = []
        for exp in os.listdir(dir):
            for file in os.listdir(os.path.join(dir, exp, 'x')):
                xy = [os.path.join(dir, exp, 'x', file), os.path.join(dir, exp, 'y', file)]
                self.files.append(xy)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, y = self.files[i]
        x, y = np.load(x), np.load(y)

        return torch.tensor(x).float(), torch.tensor(y).float()



# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        # check if we are beyond the dataset
        if end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1]
        X.append(seq_x)
        y.append(seq_y)

    return X, y

def create_dataset(filename: str) -> pd.DataFrame:
    # Initialize the DataFrame with exposure, temperature, and humidity columns
    df = pd.DataFrame()

    # Load and process the S0 sheet separately for initial exposure, temperature, and humidity
    S0_db = pd.read_excel(filename, sheet_name="S0")
    S0_db = S0_db.infer_objects()
    if sum(S0_db["Exposure"]) > 0:
        end_idx = S0_db[S0_db["Exposure"] == 1]["Routine Counter"].iloc[0] + POSITIVE_LABEL_TIME
        S0_db.drop(S0_db[S0_db["Routine Counter"] > end_idx].index, inplace=True)
    S0_filtered = S0_db[S0_db["Routine Counter"] >= WARMUP]
    df = S0_filtered[["Exposure", "Temperature", "Humidity"]].join(df)

    for i in range(0, 24):  # Adjust the range if the number of sheets is different
        sheet_name = f"S{i}"
        # Read the sheet into a DataFrame
        sheet_df = pd.read_excel(filename, sheet_name=sheet_name)
        # Infer objects and filter based on "Routine Counter"
        sheet_df = sheet_df.infer_objects()
        if sum(sheet_df ["Exposure"]) > 0:
            sheet_df.drop(sheet_df[sheet_df["Routine Counter"] > end_idx].index, inplace=True)
        filtered_df = sheet_df[sheet_df["Routine Counter"] >= WARMUP]
        # Extract the relevant column and assign it to the main DataFrame
        df[f"S{i}"] = filtered_df[PARAMETER]
    
    return df

if __name__ == '__main__':
    # Train
    training_data_files = []
    training_dir = "data/train/"
    for file in os.listdir(training_dir):
        training_data_files.append(os.path.join(training_dir, file))

    r = process_map(create_dataset, training_data_files[:3])

    filtered_data = "data_filtered"
    os.mkdir(filtered_data)
    os.mkdir(os.path.join(filtered_data, "train"))

    dd = pd.concat(r)
    dd.replace(-999, 0, inplace=True)
    MM = StandardScaler()
    MM.fit(dd[SENSORS])
    joblib.dump(MM, MM_FILE_NAME)
    dd = None
    
    for idx, df in enumerate(r):
        base_dir = os.path.join(filtered_data, "train", str(idx))
        os.mkdir(base_dir)
        x_dir = os.path.join(base_dir, 'x')
        os.mkdir(x_dir)
        y_dir = os.path.join(base_dir, 'y')
        os.mkdir(y_dir)

        df.replace(-999, 0, inplace=True)
        df[SENSORS] = MM.transform(df[SENSORS])

        X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
        assert len(X) == len(y)
        X_ss, y_mm = split_sequences(X, y, LENGTH)

        for i in range(len(X_ss)):
            in_out = zip(X_ss, y_mm)
            np.save(os.path.join(x_dir, f"{str(i)}.npy"), np.array(X_ss[i]))
            np.save(os.path.join(y_dir, f"{str(i)}.npy"), np.array(y_mm[i]))


    # Validation
    val_data_files = []
    val_dir = "data/val/"
    for file in os.listdir(val_dir):
        val_data_files.append(os.path.join(val_dir, file))

    r = process_map(create_dataset, val_data_files)

    os.mkdir(os.path.join(filtered_data, "val"))

    for idx, df in enumerate(r):
        base_dir = os.path.join(filtered_data, "val", str(idx))
        os.mkdir(base_dir)
        x_dir = os.path.join(base_dir, 'x')
        os.mkdir(x_dir)
        y_dir = os.path.join(base_dir, 'y')
        os.mkdir(y_dir)

        df.replace(-999, 0, inplace=True)
        df[SENSORS] = MM.transform(df[SENSORS])

        X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
        assert len(X) == len(y)
        X_ss, y_mm = split_sequences(X, y, LENGTH)

        for i in range(len(X_ss)):
            np.save(os.path.join(x_dir, f"{str(i)}.npy"), np.array(X_ss[i]))
            np.save(os.path.join(y_dir, f"{str(i)}.npy"), np.array(y_mm[i]))

    # Testing
    test_data_files = []
    test_dir = "data/test/"
    for file in os.listdir(test_dir):
        test_data_files.append(os.path.join(test_dir, file))

    r = process_map(create_dataset, test_data_files)

    os.mkdir(os.path.join(filtered_data, "test"))

    for idx, df in enumerate(r):
        base_dir = os.path.join(filtered_data, "test", str(idx))
        os.mkdir(base_dir)
        x_dir = os.path.join(base_dir, 'x')
        os.mkdir(x_dir)
        y_dir = os.path.join(base_dir, 'y')
        os.mkdir(y_dir)

        df.replace(-999, 0, inplace=True)
        df[SENSORS] = MM.transform(df[SENSORS])

        X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
        assert len(X) == len(y)
        X_ss, y_mm = split_sequences(X, y, LENGTH)

        for i in range(len(X_ss)):
            np.save(os.path.join(x_dir, f"{str(i)}.npy"), np.array(X_ss[i]))
            np.save(os.path.join(y_dir, f"{str(i)}.npy"), np.array(y_mm[i]))
