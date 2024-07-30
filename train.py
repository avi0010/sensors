import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from simple_lstm import ShallowRegressionGRU, ShallowRegressionLSTM
from lstm_fcn import MLSTMfcn
from dataset import SequenceDataset, create_dataset

LEARNING_RATE = 0.01
LEN_FEATURES = 26
EPOCHS = 5
THRESHOLD = 0.5
PATIENCE = 5
PATIENCE_FACTOR = 0.1
MODEL_BASE_PATH = "training"
HIDDEN_LAYERS = 2
NUM_RNN_LAYERS = 1
FEATURE_LENGTH = 100
MODEL = "GRU"
BASE_DIR = "data"
MODEL_SAVE_PATH = os.path.join(MODEL_BASE_PATH, f"{MODEL}_{HIDDEN_LAYERS}_{NUM_RNN_LAYERS}_{FEATURE_LENGTH}")

if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

if MODEL == "GRU":
    model = ShallowRegressionGRU(LEN_FEATURES, HIDDEN_LAYERS, NUM_RNN_LAYERS)
elif MODEL == "LSTM":
    model = ShallowRegressionLSTM(LEN_FEATURES, HIDDEN_LAYERS, NUM_RNN_LAYERS)
elif MODEL == "FCN_LSTM":
    model = MLSTMfcn(num_classes=1, num_features=LEN_FEATURES)
else:
    raise ValueError(f"{MODEL} not implemented")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=PATIENCE_FACTOR)

dfs = []
for file in os.listdir(os.path.join(BASE_DIR, "training")):
    df = create_dataset(os.path.join(BASE_DIR, file), True)
    dfs.append(df)

train_dataset = SequenceDataset(dfs, mode="train", length=FEATURE_LENGTH)

dfs_val = []
for file in os.listdir(os.path.join(BASE_DIR, "val")):
    df = create_dataset(os.path.join(BASE_DIR, file), True)
    dfs_val.append(df)

val_dataset = SequenceDataset(dfs_val, mode="val", length=FEATURE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

v_loss, t_loss = [], []
v_accuracy     = []
best_vloss = 10000000

for epoch in range(EPOCHS):
    running_loss = 0.

    model.train()

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = (model(inputs)).squeeze(-1)

        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
    t_loss.append(running_loss / len(train_loader))

    model.eval()

    correct, total = 0, 0
    running_v_loss = 0.0
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            inputs, labels = data
            outputs = model(inputs).squeeze(-1)
            loss = loss_function(outputs, labels)
            running_v_loss += loss.item()

            outputs.sigmoid_()
            outputs[outputs < THRESHOLD] = 0.0
            outputs[outputs > THRESHOLD] = 1.0
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

    avg_vloss = running_v_loss / len(val_loader)
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(MODEL_SAVE_PATH, f"{epoch + 1}_{round(avg_vloss, 4)}.pth")
        torch.save(model, model_path)

    v_loss.append(avg_vloss)
    v_accuracy.append(correct / total)

xs = [x for x in range(EPOCHS)]

f, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(xs, t_loss, label="t_loss")
ax[0].plot(xs, v_loss, "-.", label="v_loss")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[0].set_title("Model loss")

ax[1].plot(xs, v_accuracy, label="acc")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("acc")
ax[1].legend()
ax[1].set_title("Model acc")

f.savefig(os.path.join(MODEL_SAVE_PATH, "fig.png"))
