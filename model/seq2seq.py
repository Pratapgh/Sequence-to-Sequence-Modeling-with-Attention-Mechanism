
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size, num_layers)

    def forward(self, source, target):
        encoder_outputs, hidden, cell = self.encoder(source)
        batch_size = source.size(0)
        target_len = target.size(1)
        output_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, target_len, output_size).to(source.device)

        input_token = target[:, 0]  # Initialize with the start token
        for t in range(1, target_len):  # Start from the second token
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            input_token = output.argmax(dim=-1)  # Get the next token

        return outputs
