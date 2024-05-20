from typing import Optional, Type
from datetime import date, timedelta
import json
import os

from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch

from networks import LinearNetwork

PREDICTION_LENGTH = 1
HISTORY_LENGTH = 10
TRAINING_SPLIT = 0.7


def get_data_from_file(in_file: str = None):
    with open(in_file, "r") as f:
        training_data, training_answers, test_data, test_answers = json.load(f)
    return training_data, training_answers, test_data, test_answers


def get_data_from_yahoo(stock: str, out_file: Optional[str] = None, create_file: bool = True):
    print("Obtaining stock data")
    yesterday = date.today() - timedelta(1)
    stock_data = yf.download(stock, '2016-04-01', yesterday)
    # use sliding window to obtain data in 1000 day intervals
    data = []
    answers = []
    for i in range(len(stock_data) - HISTORY_LENGTH - PREDICTION_LENGTH):
        subdata = stock_data[i:i + HISTORY_LENGTH]
        answer = stock_data[i + HISTORY_LENGTH:i + HISTORY_LENGTH + PREDICTION_LENGTH]
        data.append(subdata)
        answers.append(answer)

    print("Splitting into training and test data")
    training_data = data[:int(len(data) * TRAINING_SPLIT)]
    training_answers = answers[:int(len(data) * TRAINING_SPLIT)]
    test_data = data[int(len(data) * TRAINING_SPLIT):]
    test_answers = answers[int(len(data) * TRAINING_SPLIT):]

    if not create_file:
        return training_data, training_answers, test_data, test_answers

    if not out_file:
        out_file = f"{stock}_data.txt"

    print("Outputting into file")
    # clear any old data
    open(out_file, 'w').close()
    with open(out_file, "w") as f:
        d = [training_data, training_answers, test_data, test_answers]
        json.dump(d, f, indent=2)

    return training_data, training_answers, test_data, test_answers


def train(training_data: torch.Tensor, training_answers: torch.Tensor, device: str, network: Type[torch.nn.Module]):
    print("Initializing model")
    model = network(HISTORY_LENGTH, PREDICTION_LENGTH).to(device=device)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    print("Start testing")
    i = 0
    for _ in range(5):
        # TODO only works for one day output
        for data, answer in zip(training_data, training_answers):
            optimizer.zero_grad()
            output = model(data)
            a = torch.Tensor([answer[0][0]])
            loss = loss_fn(output, a)
            if i % 50 == 0:
                print(loss.item())
            loss.backward()
            optimizer.step()
            i += 1

    return model


def test(model, test_data, test_answers):
    error = []
    difference = []
    loss = []
    loss_fn = torch.nn.MSELoss(reduction="sum")

    for data, answer in zip(test_data, test_answers):
        output = model(data)
        l = loss_fn(output, torch.Tensor(answer[0][0]))
        loss.append(l.item())
        difference.append(abs(answer[0][0].item() - output.item()))
        error.append(abs(answer[0][0].item() - output.item()) / answer[0][0].item())

    avg_error = sum(error) / len(error)
    avg_difference = sum(difference) / len(difference)
    avg_loss = sum(loss) / len(loss)
    print("avg error " + str(avg_error))
    print("avg loss " + str(avg_loss))
    print("avg difference " + str(avg_difference))


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    s = 'AAPL'
    if os.path.exists(f"data/{s}_data.txt"):
        print("Retrieving data from file")
        training_data, training_answers, test_data, test_answers = get_data_from_file(f"data/{s}_data.txt")
    else:
        print("Retrieving data from yfinance")
        training_data, training_answers, test_data, test_answers = get_data_from_yahoo(s, f"data/{s}_data.txt")
    training_data = torch.Tensor(training_data, device=device)
    training_answers = torch.Tensor(training_answers, device=device)
    test_data = torch.Tensor(test_data, device=device)
    test_answers = torch.Tensor(test_answers, device=device)

    model = train(training_data, training_answers, device, LinearNetwork)
    test(model, test_data, test_answers)


if __name__ == "__main__":
    main()
