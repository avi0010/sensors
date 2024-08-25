import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.elu = nn.ELU()

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.fc_1 = nn.Linear(in_features=self.hidden_units, out_features=128)
        self.fc_2 = nn.Linear(128, 64) # fully connected last layer
        self.fc_3 = nn.Linear(64, 1) # fully connected last layer

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc_1(hn[-1])
        out = self.elu(out)
        out = self.fc_2(out) # first dense
        out = self.elu(out) # relu
        out = self.fc_3(out) # final output
        out = self.elu(out) # relu
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
            num_layers=self.num_layers
        )

        self.fc_1 = nn.Linear(in_features=self.hidden_units, out_features=128)
        self.fc_2 = nn.Linear(128, 64) # fully connected last layer
        self.fc_3 = nn.Linear(64, 1) # fully connected last layer

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.elu(hn[-1])
        out = self.fc_1(out)
        out = self.elu(out)
        out = self.fc_2(out) # first dense
        out = self.elu(out) # relu
        out = self.fc_3(out) # final output
        out = self.elu(out) # relu

        return out

if __name__ == '__main__':
    le = LabelEncoder()
    outputs = [1,0 ,2 ,1, 0, 0, 1, 2]
    le.fit([1, 2, 1, 0,0,0])
    model = ShallowRegressionLSTM(26, 8, 1)
    print(model)
    input = torch.randn(8, 100, 26)
    with torch.no_grad():
        out = model(input)
        print(out)
