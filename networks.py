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
    def __init__(self, history_length, prediction_time):
        super().__init__()
        self.soft_max = nn.Softmax(0)
        self.linear_stack = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(history_length * 6, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 10),
            nn.Linear(10, prediction_time)
        )

    def forward(self, x):
        print("start", end=" ")
        print(x)
        x = self.soft_max(x)
        print(x)
        logits = self.linear_stack(x)
        return logits
