import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implement the PE function
    The d_model parameter represents the dimensionality of the model.
    The dropout parameter is the dropout probability used in positional encoding.
    The max_len parameter determines the maximum length of the positional encoding.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        #The div_term tensor calculates the exponential term for the positional encoding formula.
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #x is added elementwise to a portion of the positional encoding tensor self.pe.
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, hidden, num_layers, transformer_dim=32):
        super(TimeSeriesTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model=transformer_dim, max_len=100)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dim_feedforward=hidden, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(transformer_dim, 1)
        self.transformation = nn.Linear(input_dim, transformer_dim)

    def forward(self, x):
        x = self.transformation(x)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.encoder(x)  # Shape: (batch_size, seq_len, input_dim)
        x = x.mean(dim=1)  # Average pooling over the sequence length
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        return x

if __name__ == '__main__':
    model = TimeSeriesTransformer(27, 4, 32, 1)
    input = torch.randn(102, 100, 27)
    with torch.no_grad():
        forcast = model(input)
        print(forcast.shape)
