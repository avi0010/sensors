import torch
import torch.nn as nn
from trans import PositionalEncoding, LearnablePositionalEncoding
import torch.nn.functional as F
from dataset_v2 import LENGTH

class Conv_Attn_Mean(nn.Module):
    def __init__(self, input_dim, n_heads, hidden, num_layers, transformer_dim=16):
        super(Conv_Attn_Mean, self).__init__()

        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dim_feedforward=hidden, batch_first=True),
            num_layers=num_layers)

        self.channel_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dim_feedforward=hidden, batch_first=True),
            num_layers=num_layers)

        self.embedding_channel = nn.Linear(input_dim, transformer_dim)
        self.embedding_input = nn.Linear(LENGTH, transformer_dim)
        self.positional_encoding = LearnablePositionalEncoding(d_model=transformer_dim, max_len=100)
        self.gelu = nn.GELU()
        self.gate = nn.Linear(2 * transformer_dim, 2)
        self.output_linear = nn.Linear(2 * transformer_dim, 1)

    def forward(self, x):
        temporal_attn = self.gelu(self.embedding_channel(x))
        temporal_attn = self.positional_encoding(temporal_attn)  # Add positional encoding
        temporal_attn = self.temporal_attn(temporal_attn)  # Shape: (batch_size, seq_len, input_dim)
        temporal_attn = temporal_attn.mean(dim=1)

        channel_attn = self.embedding_input(x.transpose(-1, -2))
        channel_attn = self.channel_attn(channel_attn)  # Shape: (batch_size, seq_len, input_dim)
        channel_attn = channel_attn.mean(dim=1)

        gate = F.softmax(self.gate(torch.cat([temporal_attn, channel_attn], dim=-1)), dim=-1)

        encoding = torch.cat([temporal_attn * gate[:, 0:1], channel_attn * gate[:, 1:2]], dim=-1)

        output = self.output_linear(encoding)
        return output

if __name__ == '__main__':
    model = Conv_Attn_Mean(27, 2, 32, 1)
    input = torch.randn(64, 100, 27)
    with torch.no_grad():
        forcast = model(input)
        print(forcast.shape)
