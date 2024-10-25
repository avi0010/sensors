import torch
from torch.utils.data import Dataset
import pandas as pd
from dataset import create_dataset, MM_FILE_NAME, split_sequences, SENSORS
import numpy as np
from simple_lstm import ShallowRegressionLSTM, ShallowRegressionGRU
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib

model = {
    "Model1": "./training/FCN_LSTM_16_2_100_289cd27c-8e5d-439f-bd59-b6f69bbc0d93/5_0.2125.pth",
    "Model2": "./training/FCN_LSTM_16_2_100_ef6673be-0252-469e-86f3-3aeec9c74fb5/10_0.1389.pth",
    "Model3": "./training/FCN_LSTM_16_2_100_f10740c0-708f-44c4-9f30-35501906c1c2/5_0.2639.pth",
    "Model4": "./training/LSTM_16_1_100_05313d56-2cc6-4f41-b453-e48b47bfde9a/7_0.1483.pth",
    "Model5": "./training/LSTM_16_1_100_6db4c3c3-8562-4576-b8ef-d77022709259/17_0.3128.pth",
    "Model6": "./training/LSTM_16_1_100_90115625-d2c9-4abf-9d6c-ce2067edae0b/16_0.2525.pth",
    "Model7": "./training/LSTM_16_2_100_2b690aae-d337-4008-a8b4-66fa67b57bf3/26_0.2809.pth",
    "Model8": "./training/LSTM_16_2_100_ef747907-af1f-4b8b-9454-2e77dc658bcd/34_0.1192.pth",
    "Model9": "./training/LSTM_16_2_100_f610466d-4c72-4996-97ee-4d358acab3c1/11_0.21.pth"
}

BASE_DIR = "./data/"


class SequenceDataset(Dataset):
    def __init__(self, dataframes, length=100):
        dd = pd.concat(dataframes)
        dd.replace(-999, 0, inplace=True)
        MM = joblib.load(MM_FILE_NAME)

        self.train_x, self.train_y = [], []

        for df in dataframes:
            df.replace(-999, 0, inplace=True)
            df[SENSORS] = MM.transform(df[SENSORS])
            # fig = px.line(df, template="plotly_dark")
            # fig.show()
            X, y = df.drop(columns=['Exposure']).to_numpy(), df.Exposure.values
            X_ss, y_mm = split_sequences(X, y, length)
            self.train_x.extend(X_ss)
            self.train_y.extend(y_mm)

        assert len(self.train_x) == len(self.train_y)

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, i):
        return torch.tensor(self.train_x[i]).float(), torch.tensor(self.train_y[i]).float()

for dic in ["val", "test"]:
    for enu, file in tqdm(enumerate(os.listdir(os.path.join(BASE_DIR, dic)))):
        print(os.path.join(BASE_DIR, dic, file))
        df = create_dataset(os.path.join(BASE_DIR, dic, file))
        test_dataset = SequenceDataset([df], length=100)
        train_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False)
        
        for mod, path in model.items():
            MODEL = torch.load(path)

            pred = []
            true = []

            MODEL.eval()

            for i, data in enumerate(train_loader):
                with torch.no_grad():
                    inputs, labels = data
                    inputs = inputs.to('cuda')
                    outputs = MODEL(inputs).squeeze(-1).to("cpu")
                    outputs.sigmoid_()
                    # outputs[outputs < THRESHOLD] = 0.0
                    # outputs[outputs >= THRESHOLD] = 1.0
                    pred.extend(outputs.tolist())
                    true.extend(labels.tolist())

            res_dic = {"label": true, "predictions": pred}
            res_df = pd.DataFrame(res_dic)

            with pd.ExcelWriter(os.path.join(BASE_DIR, dic, file), mode='a') as writer:  
                res_df.to_excel(writer, sheet_name=mod)
