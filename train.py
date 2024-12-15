import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from simple_lstm import ShallowRegressionGRU, ShallowRegressionLSTM
from lstm_fcn import MLSTMfcn
from conv_attn import Conv_Attn
from graph import graph_model 
from dataset import SequenceDataset
from dataset_v2 import LENGTH
from create import INPUTS, PARAMETER
from tqdm import tqdm
from trans import TimeSeriesTransformer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import json
import uuid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 0.0003
LEN_FEATURES = len(INPUTS)
EPOCHS = 100
THRESHOLD = 0.5
PATIENCE = 5
PATIENCE_FACTOR = 0.1
MODEL_BASE_PATH = "training"
HIDDEN_LAYERS = 32
NUM_RNN_LAYERS = 1
FEATURE_LENGTH = LENGTH
POS_WEIGHT = 3
HEADS = 4
MODEL = "trans"
BASE_DIR = "data"
GAMMA = 0.95
MODEL_SAVE_PATH = os.path.join(MODEL_BASE_PATH, f"{MODEL}_{HIDDEN_LAYERS}_{NUM_RNN_LAYERS}_{FEATURE_LENGTH}_{str(uuid.uuid4())}")
print(MODEL_SAVE_PATH)

if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)


with open (os.path.join(MODEL_SAVE_PATH, "parameters.json"), 'w') as f:
   data = {
       "LEARNING_RATE": LEARNING_RATE,
       "PARAMETER" :PARAMETER ,
       "FEATURE_LENGTH" :FEATURE_LENGTH ,
       "LEN_FEATURES" :LEN_FEATURES ,
       "MODEL" :MODEL ,
       "GAMMA": GAMMA ,
       "POS_WEIGHT": POS_WEIGHT,
       "NUM_RNN_LAYERS" :NUM_RNN_LAYERS ,
       "PATIENCE_FACTOR" :PATIENCE_FACTOR ,
       "PATIENCE" :PATIENCE ,
       "THRESHOLD" :THRESHOLD ,
       "EPOCHS" :EPOCHS ,
       "HEADS" :HEADS ,
   }
   print(data)
   json.dump(data, f, indent=4)

if MODEL == "GRU":
    model = ShallowRegressionGRU(LEN_FEATURES, HIDDEN_LAYERS, NUM_RNN_LAYERS).to(DEVICE)
elif MODEL == "LSTM":
    model = ShallowRegressionLSTM(LEN_FEATURES, HIDDEN_LAYERS, NUM_RNN_LAYERS).to(DEVICE)
elif MODEL == "FCN_LSTM":
    model = MLSTMfcn(num_classes=1, num_features=LEN_FEATURES).to(DEVICE)
elif MODEL == "graph":
    model = graph_model(LEN_FEATURES, 2, LENGTH, 1).to(DEVICE)
elif MODEL == "trans":
    model = TimeSeriesTransformer(input_dim=LEN_FEATURES, n_heads=HEADS, hidden=HIDDEN_LAYERS, num_layers=NUM_RNN_LAYERS).to(DEVICE)
elif MODEL == "conv_attn":
    model = Conv_Attn(input_dim=LEN_FEATURES, n_heads=HEADS, hidden=HIDDEN_LAYERS, num_layers=NUM_RNN_LAYERS).to(DEVICE)
else:
    raise ValueError(f"{MODEL} not implemented")

loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

train_dataset = SequenceDataset("./data_filtered/train/")

val_dataset = SequenceDataset("./data_filtered/val/")

train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16384, shuffle=True, num_workers=4)

v_loss, t_loss = [], []
v_accuracy     = []
v_precision, v_recall, v_f1 = [], [], []
t_precision, t_recall, t_f1 = [], [], []
best_vloss = 10000000

for epoch in tqdm(range(EPOCHS)):
    running_loss = 0.

    model.train()

    true_labels, preds = [], []
    for i, data in tqdm(enumerate(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = (model(inputs)).squeeze(-1)

        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        loss.backward()
        running_loss += loss.to("cpu").item()

        # Adjust learning weights
        optimizer.step()

        outputs.sigmoid_()
        outputs[outputs < THRESHOLD] = 0.0
        outputs[outputs > THRESHOLD] = 1.0
        labels = labels.detach().to("cpu", dtype=torch.uint8)
        outputs = outputs.detach().to("cpu", dtype=torch.uint8)
        true_labels.extend(labels.tolist())
        preds.extend(outputs.tolist())

        # Gather data and report
    t_loss.append(running_loss / len(train_loader))
    precision, recall, f1 = precision_score(true_labels, preds), recall_score(true_labels, preds), f1_score(true_labels, preds)
    accuracy = accuracy_score(true_labels, preds)
    t_precision.append(precision)
    t_recall.append(recall)
    t_f1.append(f1)

    model.eval()

    correct, total = 0, 0
    running_v_loss = 0.0
    true_labels, preds = [], []
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).squeeze(-1)
            loss = loss_function(outputs, labels)
            running_v_loss += loss.to("cpu").item()

            outputs.sigmoid_()
            outputs[outputs < THRESHOLD] = 0.0
            outputs[outputs > THRESHOLD] = 1.0
            labels = labels.to("cpu", dtype=torch.uint8)
            outputs = outputs.to("cpu", dtype=torch.uint8)
            true_labels.extend(labels.tolist())
            preds.extend(outputs.tolist())

    precision, recall, f1 = precision_score(true_labels, preds), recall_score(true_labels, preds), f1_score(true_labels, preds)
    accuracy = accuracy_score(true_labels, preds)
    v_precision.append(precision)
    v_recall.append(recall)
    v_f1.append(f1)
    avg_vloss = running_v_loss / len(val_loader)
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
    model_path = os.path.join(MODEL_SAVE_PATH, f"{epoch + 1}_{round(avg_vloss, 4)}.pth")
    torch.save(model, model_path)

    v_loss.append(avg_vloss)
    v_accuracy.append(accuracy)
    scheduler.step()

xs = [x for x in range(EPOCHS)]

f, ax = plt.subplots(1, 3, figsize=(15, 5))

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

ax[2].plot(xs, v_precision, label="v_precision")
ax[2].plot(xs, v_recall, label="v_recall")
ax[2].plot(xs, v_f1, label="v_f1")
ax[2].plot(xs, t_precision, label="t_precision")
ax[2].plot(xs, t_recall, label="t_recall")
ax[2].plot(xs, t_f1, label="t_f1")
ax[2].set_xlabel("epoch")
ax[2].set_ylabel("stats")
ax[2].legend()
ax[2].set_title("Model stats")
f.savefig(os.path.join(MODEL_SAVE_PATH, "fig.png"))
