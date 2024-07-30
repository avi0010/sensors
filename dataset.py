import torch
from torch.utils.data import Dataset
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
SENSORS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "Temperature", "Humidity", ]
SENSOR_MIN = 0
SENSOR_MAX = 2
SENSOR_NEW_MIN = 0
SENSOR_NEW_MAX = 1

MM_FILE_NAME = "scaler.gz"

class SequenceDataset(Dataset):
    def __init__(self, dataframes, mode="train"):
        dd = pd.concat(dataframes)
        dd.replace(-999, 0, inplace=True)
        if mode == "train":
            MM = MinMaxScaler()
            MM.fit(dd[SENSORS])
            joblib.dump(MM, MM_FILE_NAME)
        elif mode == "val" or mode == "test":
            MM = joblib.load(MM_FILE_NAME)
        else:
            raise ValueError(f"{mode} is not valid")

        self.train_x, self.train_y = [], []

        for df in dataframes:
            df[SENSORS] = MM.transform(df[SENSORS])
            X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
            X_ss, y_mm = split_sequences(X, y, 100)
            self.train_x.extend(X_ss)
            self.train_y.extend(y_mm)

        assert len(self.train_x) == len(self.train_y)

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, i):
        return torch.tensor(self.train_x[i]).float(), torch.tensor(self.train_y[i]).float()



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

def create_dataset(filename, if_split=True):
    S0_db = pd.read_excel(filename, sheet_name="S0")
    S0_db = S0_db.infer_objects()
    S0_filtered = S0_db[S0_db["Routine Counter"] >= 1]
    df = S0_filtered[["Exposure", "Temperature", "Humidity"]]

    S0 = S0_filtered["Raw"]
    df = df.assign(S0=S0)

    S1_db = pd.read_excel(filename, sheet_name="S1")
    S1_db = S1_db.infer_objects()
    S1_filtered = S1_db[S1_db["Routine Counter"] >= 1]
    S1 = S1_filtered["Raw"]
    df = df.assign(S1=S1)

    S2_db = pd.read_excel(filename, sheet_name="S2")
    S2_db = S2_db.infer_objects()
    S2_filtered = S2_db[S2_db["Routine Counter"] >= 1]
    S2 = S2_filtered["Raw"]
    df = df.assign(S2=S2)

    S3_db = pd.read_excel(filename, sheet_name="S3")
    S3_db = S3_db.infer_objects()
    S3_filtered = S3_db[S3_db["Routine Counter"] >= 1]
    S3 = S3_filtered["Raw"]
    df = df.assign(S3=S3)

    S4_db = pd.read_excel(filename, sheet_name="S4")
    S4_db = S4_db.infer_objects()
    S4_filtered = S4_db[S4_db["Routine Counter"] >= 1]
    S4 = S4_filtered["Raw"]
    df = df.assign(S4=S4)

    S5_db = pd.read_excel(filename, sheet_name="S5")
    S5_db = S5_db.infer_objects()
    S5_filtered = S5_db[S5_db["Routine Counter"] >= 1]
    S5 = S5_filtered["Raw"]
    df = df.assign(S5=S5)

    S6_db = pd.read_excel(filename, sheet_name="S6")
    S6_db = S6_db.infer_objects()
    S6_filtered = S6_db[S6_db["Routine Counter"] >= 1]
    S6 = S6_filtered["Raw"]
    df = df.assign(S6=S6)

    S7_db = pd.read_excel(filename, sheet_name="S7")
    S7_db = S7_db.infer_objects()
    S7_filtered = S7_db[S7_db["Routine Counter"] >= 1]
    S7 = S7_filtered["Raw"]
    df = df.assign(S7=S7)

    S8_db = pd.read_excel(filename, sheet_name="S8")
    S8_db = S8_db.infer_objects()
    S8_filtered = S8_db[S8_db["Routine Counter"] >= 1]
    S8 = S8_filtered["Raw"]
    df = df.assign(S8=S8)

    S9_db = pd.read_excel(filename, sheet_name="S9")
    S9_db = S9_db.infer_objects()
    S9_filtered = S9_db[S9_db["Routine Counter"] >= 1]
    S9 = S9_filtered["Raw"]
    df = df.assign(S9=S9)

    S10_db = pd.read_excel(filename, sheet_name="S10")
    S10_db = S10_db.infer_objects()
    S10_filtered = S10_db[S10_db["Routine Counter"] >= 1]
    S10 = S10_filtered["Raw"]
    df = df.assign(S10=S10)

    S11_db = pd.read_excel(filename, sheet_name="S11")
    S11_db = S11_db.infer_objects()
    S11_filtered = S11_db[S11_db["Routine Counter"] >= 1]
    S11 = S11_filtered["Raw"]
    df = df.assign(S11=S11)

    S12_db = pd.read_excel(filename, sheet_name="S12")
    S12_db = S12_db.infer_objects()
    S12_filtered = S12_db[S12_db["Routine Counter"] >= 1]
    S12 = S12_filtered["Raw"]
    df = df.assign(S12=S12)

    S13_db = pd.read_excel(filename, sheet_name="S13")
    S13_db = S13_db.infer_objects()
    S13_filtered = S13_db[S13_db["Routine Counter"] >= 1]
    S13 = S13_filtered["Raw"]
    df = df.assign(S13=S13)

    S14_db = pd.read_excel(filename, sheet_name="S14")
    S14_db = S14_db.infer_objects()
    S14_filtered = S14_db[S14_db["Routine Counter"] >= 1]
    S14 = S14_filtered["Raw"]
    df = df.assign(S14=S14)

    S15_db = pd.read_excel(filename, sheet_name="S15")
    S15_db = S15_db.infer_objects()
    S15_filtered = S15_db[S15_db["Routine Counter"] >= 1]
    S15 = S15_filtered["Raw"]
    df = df.assign(S15=S15)

    S16_db = pd.read_excel(filename, sheet_name="S16")
    S16_db = S16_db.infer_objects()
    S16_filtered = S16_db[S16_db["Routine Counter"] >= 1]
    S16 = S16_filtered["Raw"]
    df = df.assign(S16=S16)

    S17_db = pd.read_excel(filename, sheet_name="S17")
    S17_db = S17_db.infer_objects()
    S17_filtered = S17_db[S17_db["Routine Counter"] >= 1]
    S17 = S17_filtered["Raw"]
    df = df.assign(S17=S17)

    S18_db = pd.read_excel(filename, sheet_name="S18")
    S18_db = S18_db.infer_objects()
    S18_filtered = S18_db[S18_db["Routine Counter"] >= 1]
    S18 = S18_filtered["Raw"]
    df = df.assign(S18=S18)

    S19_db = pd.read_excel(filename, sheet_name="S19")
    S19_db = S19_db.infer_objects()
    S19_filtered = S19_db[S19_db["Routine Counter"] >= 1]
    S19 = S19_filtered["Raw"]
    df = df.assign(S19=S19)

    S20_db = pd.read_excel(filename, sheet_name="S20")
    S20_db = S20_db.infer_objects()
    S20_filtered = S20_db[S20_db["Routine Counter"] >= 1]
    S20 = S20_filtered["Raw"]
    df = df.assign(S20=S20)

    S21_db = pd.read_excel(filename, sheet_name="S21")
    S21_db = S21_db.infer_objects()
    S21_filtered = S21_db[S21_db["Routine Counter"] >= 1]
    S21 = S21_filtered["Raw"]
    df = df.assign(S21=S21)

    S22_db = pd.read_excel(filename, sheet_name="S22")
    S22_db = S22_db.infer_objects()
    S22_filtered = S22_db[S22_db["Routine Counter"] >= 1]
    S22 = S22_filtered["Raw"]
    df = df.assign(S22=S22)

    S23_db = pd.read_excel(filename, sheet_name="S23")
    S23_db = S23_db.infer_objects()
    S23_filtered = S23_db[S23_db["Routine Counter"] >= 1]
    S23 = S23_filtered["Raw"]
    df = df.assign(S23=S23)

    # df.replace(-999, 0, inplace=True)
    # X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
    # X_ss, y_mm = split_sequences(X, y, 100)
    return df

if __name__ == '__main__':
    base_dir = "/home/aveekal/Downloads/OneDrive_2024-07-26/41X-Post Processed/Gas_Testing/"
    train_x, train_y, dfs = [], [], []
    for file in os.listdir(base_dir)[:1]:
        print(file)
        df = create_dataset(os.path.join(base_dir, file), True)
        # train_x.extend(x)
        # train_y.extend(y)
        dfs.append(df)

    train_dataset = SequenceDataset(dfs)

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
