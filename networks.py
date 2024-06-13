import torch
from torch import nn

"""
Ideas
can use today's date and see if that effects the quality of the output
can use actions=True for yfinance for dividends and stock splits

use bilinear to make input layer smaller
use pairwisedistance to get a day to day change
use convolution or something to make the 6 different values into one

start with simple linear layers and add different activation layers in between
then try using bilinear
then try implmenting pairwise to both

use three years worth of data for predictions
expirement with different output sizes (1 day vs 1 week)
use multiple nn to get different information one for each of the following:
    - open
    - high
    - low
    - close
    - adj close
    - volume
can obtain each of these to add to the list of prices for input and remove oldest price to keep same size
then preform in a loop to obtain multiple time steps
store trained values for different stocks for future use
"""


class LinearNetwork(nn.Module):
    def __init__(self, history_length, input_size, output_size):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_length * input_size, 10),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class LSTM(nn.Module):
    def __init__(self, seq_length, input_size, hidden_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_length, batch_first=True)
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * hidden_length, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.stack(out)
        return out


if __name__ == '__main__':
    rnn = nn.LSTM(10, 20, 2, batch_first=True)
    inp = torch.ones(5, 3, 10)
    output, (hn, cn) = rnn(inp)
    print(output.size())
    flatten = nn.Flatten()
    output = flatten(output)
    print(output.size())
