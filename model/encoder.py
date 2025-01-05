
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        outputs, (hidden, cell) = self.lstm(embedded)  # [batch_size, seq_len, hidden_size], [num_layers, batch_size, hidden_size]
        return outputs, hidden, cell
