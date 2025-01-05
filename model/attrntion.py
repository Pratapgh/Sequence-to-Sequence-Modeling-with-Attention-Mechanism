
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
        attn_weights = torch.bmm(encoder_outputs, hidden.transpose(1, 2))  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights.squeeze(2), dim=-1)  # [batch_size, seq_len]
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_size]
        return context_vector
