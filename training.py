from typing import Optional, Type, Tuple
from datetime import date, timedelta
import json
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import yfinance as yf
import torch

from networks import LinearNetwork, LSTM

PREDICTION_LENGTH = 1
HISTORY_LENGTH = 1
TRAINING_SPLIT = 0.85


def get_data_from_file(in_file: str = None):
    with open(in_file, "r") as f:
        data_min, data_max, training_data, training_answers, test_data, test_answers = json.load(f)
    scaler = MinMaxScaler()
    scaler.fit([data_min, data_max])
    return scaler, training_data, training_answers, test_data, test_answers


def get_data_from_yahoo(stock: str, out_file: Optional[str] = None, random_order=False):
    print("Obtaining stock data")
    yesterday = date.today() - timedelta(1)
    scaler = MinMaxScaler()
    stock_data = yf.download(stock, '2016-04-01', yesterday)

    in_data = []
    for index, d in stock_data.iterrows():
        in_data.append([d["Open"], d["Close"], d["Open"] - d["Close"], d["High"] - d["Low"]])
    in_data = scaler.fit_transform(in_data).tolist()

    data = []
    answers = []
    for i in range(len(in_data) - HISTORY_LENGTH - PREDICTION_LENGTH):
        subdata = in_data[i:i + HISTORY_LENGTH]
        answer = in_data[i + HISTORY_LENGTH:i + HISTORY_LENGTH + PREDICTION_LENGTH]
        data.append(subdata)
        answers.append(answer)

    if random_order:
        data, answers = shuffle(data, answers)

    print("Splitting into training and test data")
    training_data = data[:int(len(data) * TRAINING_SPLIT)]
    training_answers = answers[:int(len(data) * TRAINING_SPLIT)]
    test_data = data[int(len(data) * TRAINING_SPLIT):]
    test_answers = answers[int(len(data) * TRAINING_SPLIT):]

    print("Outputting into file")
    if not out_file:
        out_file = f"{stock}_data.txt"
    open(out_file, 'w').close()  # clear any old data
    with open(out_file, "w") as f:
        json.dump([list(scaler.data_min_), list(scaler.data_max_), training_data, training_answers, test_data, test_answers], f, indent=2)

    return scaler, training_data, training_answers, test_data, test_answers


def train(training_data: torch.Tensor, training_answers: torch.Tensor, device: str, network: Type[torch.nn.Module], arg: Tuple):
    print("Initializing model")
    model = network(*arg).to(device=device)
    model.train()
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    print("Start testing")
    i = 0
    for epoch in range(3):
        print(f"Epoch: {epoch}")
        # TODO only works for one day output
        for data, answer in zip(training_data, training_answers):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, answer[0])
            # if i % 100 == 0:
            #     print(f"Loss at {i}: {loss.item()}")
            loss.backward()
            optimizer.step()
            i += 1

    model.eval()
    return model


def test(model: Type[torch.nn.Module], test_data: torch.Tensor, test_answers: torch.Tensor):
    loss = []
    error = []
    loss_fn = torch.nn.MSELoss(reduction="sum")

    for data, answer in zip(test_data, test_answers):
        output = model(data)
        loss.append(loss_fn(output, answer[0]).item())
        e = abs((output - answer)[0][0].item())
        error.append(e)

    avg_loss = sum(loss) / len(loss)
    avg_error = sum(error) / len(error)
    print(f"{avg_loss = }")
    print(f"{avg_error = }")


def graph(model: Type[torch.nn.Module], scaler: MinMaxScaler, input_data: torch.Tensor, true_values: torch.Tensor):
    scaled_predicted = []
    true_values = [i[0] for i in true_values]
    scaled_true = scaler.inverse_transform(true_values)

    for inp in input_data:
        output = model(inp)
        output = scaler.inverse_transform([output.detach().numpy()])[0]
        scaled_predicted.append(list(output))

    true = [i[0].item() for i in scaled_true]
    predicted = [i[0] for i in scaled_predicted]

    plt.figure()
    plt.plot(true, label="true")
    plt.plot(predicted, label="predicted")
    plt.legend()

    plt.figure()
    plt.scatter(true, predicted)
    plt.plot([0, max(predicted)], [0, max(predicted)], color="red")
    plt.show()


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    s = 'AAPL'
    replace = True
    if os.path.exists(f"data/{s}_data.txt" and not replace):
        print("Retrieving data from file")
        scaler, training_data, training_answers, test_data, test_answers = get_data_from_file(f"data/{s}_data.txt")
    else:
        print("Retrieving data from yfinance")
        scaler, training_data, training_answers, test_data, test_answers = get_data_from_yahoo(s, f"data/{s}_data.txt", random_order=True)
    training_data = torch.Tensor(training_data, device=device)
    training_answers = torch.Tensor(training_answers, device=device)
    test_data = torch.Tensor(test_data, device=device)
    test_answers = torch.Tensor(test_answers, device=device)

    file_path = f"{LinearNetwork.__name__}.txt"
    load_from_file = True
    if load_from_file:
        model = torch.load(file_path)
        model.eval()
    else:
        # model = train(training_data, training_answers, device, LSTM, (len(training_data[0]), len(training_data[0][0]), 10, 1))
        model = train(training_data, training_answers, device, LinearNetwork, (len(training_data[0]), len(training_data[0][0]), len(training_data[0][0])))
        torch.save(model, file_path)

    print("Test Model on training data")
    test(model, training_data, training_answers)
    print("Test Model on test data")
    test(model, test_data, test_answers)
    graph(model, scaler, training_data, training_answers)


if __name__ == "__main__":
    main()
