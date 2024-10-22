from matplotlib.pyplot import plot
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1]  # Adjust to match the size
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, n_heads, hidden, num_layers):
        super(TimeSeriesTransformer, self).__init__()

        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"

        self.positional_encoding = PositionalEncoding(input_dim, max_len=100)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.encoder(x)  # Shape: (batch_size, seq_len, input_dim)
        x = x.mean(dim=1)  # Average pooling over the sequence length
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        return x

if __name__ == '__main__':
    model = TimeSeriesTransformer(27, 3, 32, 1)
    input = torch.randn(5, 100, 27)
    with torch.no_grad():
        forcast = model(input)
        print(forcast.shape)
