from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# extract samples of the form (x_t,...,x_{t+9},y_{t+time_steps})
# to make up our dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_steps=10):
        self.X, self.y = self.create_sequences(data, time_steps)

    def create_sequences(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i:i + time_steps, :-1].values)
            y.append(data.iloc[i + time_steps, -1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        with torch.no_grad(): # no back prop here
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = loss_function(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# testing timeeeeee
def test_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad(): # no back prop here
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.tolist())
            actuals.extend(y_batch.tolist())
    
    # illustrate how the model does on the test set
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual Values", alpha=0.7)
    plt.plot(predictions, label="Predicted Values", alpha=0.7)
    plt.legend()
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.show()

if __name__ == "__main__":

    # load data
    data = pd.read_csv('data/monthly_filled.csv')

    # model initisalisation + hyperparameters
    feature_dim = data.shape[1] - 1 # final data column is output
    LSTM_block_dim = 32
    MLP_block_dim = 8
    model = LSTM_model(feature_dim, LSTM_block_dim, MLP_block_dim)

    batch_size = 32 # hyperparam to tune
    epochs = 100
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    time_steps = 10

    # form dataloaders to please PyTorch
    # 60%, 20% and 20% for training, validation and testing, repectively
    dataset = TimeSeriesDataset(data, time_steps)
    train_split = int(0.6 * len(dataset))
    val_split   = int(0.2 * len(dataset))
    test_split  = len(dataset) - train_split - val_split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_split, val_split, test_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle just for training
    val_loader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size, shuffle=False)

    train_model(model, train_loader, val_loader, loss_function, optimiser, epochs)
    test_model(model, test_loader)