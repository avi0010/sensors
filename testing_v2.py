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
MM = joblib.load("scaler_feature.gz")

model = {
    "H_2": "./training/trans_32_1_100_765d9d5c-1b6f-4814-acf5-3dc390bf7515/63_0.1371.pth",
}

os.mkdir("results")

for dic in ["train", "val"]:
    os.mkdir(f"results/{dic}")
    for enu, file in tqdm(enumerate(os.listdir(os.path.join(BASE_DIR, dic)))):
        print(os.path.join(BASE_DIR, dic, file))
        dfs = {}

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
            dataloader = DataLoader(dataset, batch_size=16384, shuffle=False, num_workers=4)

            MODEL = torch.load(path)
            MODEL.eval()
            # Perform inference
            pred, true = [], []
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to("cuda"), batch_y  # Move to GPU
                outputs = MODEL(batch_X).squeeze(-1)
                outputs.sigmoid_()
                pred.extend(outputs.cpu().tolist())
                true.extend(batch_y.cpu().tolist())

            res_dic = {"label": true, "predictions": pred}
            res_df = pd.DataFrame(res_dic)
            dfs[mod] = res_df
            # fig = res_df.plot()
            # fig.show()

        with pd.ExcelWriter(os.path.join("results", dic, file)) as writer:  
            for k,v in dfs.items():
                v.to_excel(writer, sheet_name=k)
