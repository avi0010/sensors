import torch.nn as nn
from dataset import SequenceDataset

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.elu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc_1(self.elu(hn[-1]))
        return out


class ShallowRegressionGRU(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.elu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc_1(self.elu(hn[-1]))
        return out

if __name__ == '__main__':
    train_dataset = SequenceDataset("./data_filtered/train/")
    val_dataset = SequenceDataset("./data_filtered/val/")
