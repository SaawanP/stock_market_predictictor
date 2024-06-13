from torch import nn


class LinearNetwork(nn.Module):
    def __init__(self, history_length, input_size, output_size):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(history_length * input_size, 10),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class LSTM(nn.Module):
    def __init__(self, seq_length, input_size, hidden_length, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_length, batch_first=True)
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * hidden_length, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.stack(out)
        return out
