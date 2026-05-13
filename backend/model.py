import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type="LSTM"):
        super().__init__()

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
