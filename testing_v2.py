import torch
from create import SAVE_DIR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.options.plotting.backend = "plotly"
import os
from tqdm import tqdm
import joblib

from create import SENSORS
from dataset_v2 import LENGTH

BASE_DIR = "./data/"
MM = joblib.load("/home/aveekal/Downloads/scaler_Derivative.gz")

model = {
    "Model1": "/home/aveekal/Downloads/training/trans_32_1_100_c3d51809-da5b-4b9d-a710-23d99ef33726/47_0.1224.pth"
}

for dic in ["train"]:
    for enu, file in tqdm(enumerate(os.listdir(os.path.join(BASE_DIR, dic)))):
        print(os.path.join(BASE_DIR, dic, file))

        for mod, path in model.items():
            df = pd.read_excel(os.path.join(SAVE_DIR, file))
            df = df[SENSORS]

            data = []

            for window in df.rolling(window=LENGTH):
                if len(window) < LENGTH: continue
                X, y = window.drop(columns=['Exposure']), window.Exposure.values[-1]
                window_scaled = MM.transform(X)

                data.append((torch.tensor(window_scaled).float(), torch.tensor(y).float()))

            # Unpack the data
            X_data, y_data = zip(*data)
            # Create a TensorDataset
            dataset = TensorDataset(torch.stack(X_data), torch.stack(y_data))
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

            MODEL = torch.load(path, map_location="cpu")
            MODEL.eval()
            # Perform inference
            pred, true = [], []
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to("cpu"), batch_y  # Move to GPU
                outputs = MODEL(batch_X).squeeze(-1)
                outputs.sigmoid_()
                pred.extend(outputs.cpu().tolist())
                true.extend(batch_y.cpu().tolist())

            res_dic = {"label": true, "predictions": pred}
            res_df = pd.DataFrame(res_dic)
            # fig = res_df.plot()
            # fig.show()

            with pd.ExcelWriter(os.path.join(BASE_DIR, dic, file), mode='a') as writer:  
                res_df.to_excel(writer, sheet_name=mod)
