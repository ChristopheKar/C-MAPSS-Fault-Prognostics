import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

from .gru import GRU
from .lstm import LSTM

def train(
    device, train_loader, criterion=nn.MSELoss(),
    learning_rate=0.001, batch_size=1024,
    hidden_dim=256, epochs=5, model_type="GRU"):

    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRU(device, input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTM(device, input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Check if environment is notebook
    try:
        get_ipython
        is_notebook = True
    except:
        is_notebook = False

    # Initalize progress bars depending on environment
    if (is_notebook):
        epoch_pbar = tqdm_nb(total=epochs, desc='Epochs')
        train_pbar = tqdm_nb(total=len(train_loader), desc='Training Batches')
    else:
        epoch_pbar = tqdm(
            total=epochs, desc='Epochs', ascii=True, ncols=159)
        train_pbar = tqdm(
            total=len(train_loader),
            desc='Training Batches',
            ascii=True, ncols=159)

    model.train()
    history = {'losses': [], 'times': []}
    # Start training loop
    for epoch in range(epochs):
        history['times'].append(time.perf_counter())

        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            train_pbar.set_postfix({
                'cur. total loss': avg_loss,
                'cur. batch loss':  loss.item()
            })
            train_pbar.update(1)

        history['times'][-1] = time.perf_counter() - history['times'][-1]
        history['losses'].append(avg_loss/len(train_loader))

        epoch_pbar.set_postfix({'avg. train loss': np.mean(history['losses'])})
        epoch_pbar.update(1)

    return model, history


def predict(device, model, test_x, test_y):
    model.eval()
    inp = torch.from_numpy(np.array(test_x))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)
    outputs = out.cpu().detach().numpy().reshape(-1)
    return outputs
