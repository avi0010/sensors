import os
from create import SAVE_DIR, PARAMETER, SENSORS, INPUTS
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from sklearn.preprocessing import StandardScaler
import joblib

STEP_SIZE = 2
MM_FILE_NAME = f"scaler_{PARAMETER}.gz"
LENGTH = 100

def save_dataset(file_path: str):
    file_name = file_path.split("/")[-1]
    file_dir = file_path.split("/")[-2]

    save_location = os.path.join(filtered_data, file_dir, file_name)
    os.mkdir(save_location)

    x_dir = os.path.join(save_location, 'x')
    os.mkdir(x_dir)
    y_dir = os.path.join(save_location, 'y')
    os.mkdir(y_dir)

    data_file = os.path.join(SAVE_DIR, file_name)
    df = pd.read_excel(data_file, usecols=SENSORS)

    for idx, window in enumerate(df.rolling(window=LENGTH, step=STEP_SIZE)):
        if len(window) < LENGTH: continue

        X, y = window.drop(columns=['Exposure']), window.Exposure.values[-1]
        window_scaled = MM.transform(X[INPUTS])

        np.save(os.path.join(x_dir, f"{str(idx)}.npy"), window_scaled)
        np.save(os.path.join(y_dir, f"{str(idx)}.npy"), y)

if __name__ == '__main__':
    data_dir = "data"
    filtered_data = "data_filtered"
    os.mkdir(filtered_data)

    dfs = []
    for file in os.listdir(os.path.join(data_dir, "train")):
        df = pd.read_excel(os.path.join(SAVE_DIR, file), usecols=SENSORS)
        dfs.append(df)

    dd = pd.concat(dfs)
    MM = StandardScaler()
    MM.fit(dd[INPUTS])
    joblib.dump(MM, MM_FILE_NAME)
    dd = None

    for dic in ["train", "val"]:
        os.mkdir(os.path.join(filtered_data, dic))
        files = []
        for f in os.listdir(os.path.join("data", dic)):
            files.append(os.path.join("data", dic, f))

        process_map(save_dataset, files)
