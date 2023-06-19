import torch
import torch.nn as nn


class LayerNormBasicLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormBasicLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 4, bias=False)
        self.ln_layers = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(4))
        self.cell_ln = nn.LayerNorm(hidden_size)

    def forward(self, inputs, state):
        h, c = state
        concat = self.linear(torch.cat([inputs, h], dim=1))
        i, j, f, o = torch.chunk(concat, 4, 1)
        i = torch.sigmoid(self.ln_layers[0](i))
        j = torch.tanh(self.ln_layers[1](j))
        f = torch.sigmoid(self.ln_layers[2](f))
        o = torch.sigmoid(self.ln_layers[3](o))

        new_c = self.cell_ln(i * j + (f * c))
        new_h = o * torch.tanh(new_c)

        return new_h, new_c
