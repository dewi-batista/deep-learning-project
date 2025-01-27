#=============================================================================#
#  LSTM_unemployment and LSTM_inflation_rate are the same up to hyperparams.  #
#=============================================================================#

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# architecture construction
class LSTM_model(nn.Module):
    def __init__(self, input_dim, LSTM_block_dim, MLP_block_dim):
        super(LSTM_model, self).__init__()
        self.LSTM_block_1 = nn.LSTM(input_dim, LSTM_block_dim, batch_first=True)
        self.LSTM_block_2 = nn.LSTM(LSTM_block_dim, LSTM_block_dim, batch_first=True)
        self.LSTM_block_3 = nn.LSTM(LSTM_block_dim, LSTM_block_dim, batch_first=True)
        
        self.MLP_layer_1 = nn.Linear(LSTM_block_dim, MLP_block_dim)
        self.MLP_layer_2 = nn.Linear(MLP_block_dim, MLP_block_dim)
        self.output = nn.Linear(MLP_block_dim, 1)

    def forward(self, x):
        # feed through LSTM blocks
        x, _ = self.LSTM_block_1(x)
        x, _ = self.LSTM_block_2(x)
        x, _ = self.LSTM_block_3(x)

        # feed output of final LSTM block as input to MLP
        x = x[:, -1, :]
        x = torch.relu(self.MLP_layer_1(x))
        x = torch.relu(self.MLP_layer_2(x))
        x = self.output(x)
        return x

# training timeeeeee
def train_model(model, train_loader, val_loader, loss_function, optimiser, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad() # ?
            outputs = model(X_batch)
            loss = loss_function(outputs.squeeze(), y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # compute validation loss at each epoch
        val_loss = 0.0
        model.eval()
        with torch.no_grad(): # included to make sure no param updates during validation
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = loss_function(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# testing timeeeeee
def test_model(model, test_loader):
    model.eval() # included to make sure dropout doesn't apply during testing
    preds, actuals = [], []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        preds.extend(outputs.tolist())
        actuals.extend(y_batch.tolist())
    
    # illustrate how the model does on the test set
    plt.figure()
    plt.plot(actuals, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Window Index")
    plt.ylabel("Value")
    plt.show()

# extract windows of the form (x_t,...,x_{t+h},y_{t+h+1}) to form train, test and val
class Windowify(Dataset):
    def __init__(self, data, time_steps=10):
        self.X, self.y = self.windowify(data, time_steps)

    def windowify(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i : i + time_steps, : -1].values)
            y.append(data.iloc[i + time_steps, -1])
        
        # the console whines if X and y aren't converted into floats
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":

    # load data
    data = pd.read_csv('data/monthly_filled.csv')

    # model initisalisation + hyperparameters
    covariate_dim = data.shape[1] - 1 # final column is target covariate
    LSTM_block_dim = 32
    MLP_block_dim = 8
    model = LSTM_model(covariate_dim, LSTM_block_dim, MLP_block_dim)

    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    time_steps = 10

    # sliding window timeeeeee
    dataset = Windowify(data, time_steps)

    # assign 60%, 20% and 20% for training, validation and testing, repectively
    training_split = int(0.6 * len(dataset))
    vallidation_split = int(0.2 * len(dataset))
    testing_split = int(0.2 * len(dataset))

    # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
    # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
    train_dataset = torch.utils.data.Subset(dataset, list(range(training_split)))
    val_dataset   = torch.utils.data.Subset(dataset, list(range(training_split, training_split + vallidation_split)))
    test_dataset  = torch.utils.data.Subset(dataset, list(range(training_split + vallidation_split, len(dataset))))
    
    # only shuffle training, doesn't make a difference for the other two
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset  , batch_size=batch_size)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size)

    train_model(model, train_loader, val_loader, loss_function, optimiser, epochs)
    test_model(model, test_loader)