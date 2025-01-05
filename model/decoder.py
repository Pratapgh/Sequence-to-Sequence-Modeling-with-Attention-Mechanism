
import torch
import torch.nn as nn
from models.attention import Attention

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x).unsqueeze(1)  # [batch_size, 1, hidden_size]
        context_vector = self.attention(hidden[0], encoder_outputs)  # [batch_size, hidden_size]
        lstm_input = torch.cat((embedded.squeeze(1), context_vector), dim=-1)  # [batch_size, hidden_size * 2]
        output, (hidden, cell) = self.lstm(lstm_input.unsqueeze(1), (hidden, cell))  # [batch_size, hidden_size]
        output = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
        return output, hidden, cell
