from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

# architecture construction


class MultiTask_model(nn.Module):
    def __init__(self, input_dim, LSTM_block_dim, MLP_block_dim):
        super(MultiTask_model, self).__init__()
        self.LSTM_block_1 = nn.LSTM(
            input_dim, LSTM_block_dim, batch_first=True)
        self.LSTM_block_2 = nn.LSTM(
            LSTM_block_dim, LSTM_block_dim, batch_first=True)
        self.LSTM_block_3 = nn.LSTM(
            LSTM_block_dim, LSTM_block_dim, batch_first=True)

        self.batch_norm = nn.BatchNorm1d(LSTM_block_dim)
        self.dropout = nn.Dropout(0.5)
        # splitting

        self.LSTM_block_unemp = nn.LSTM(
            LSTM_block_dim, LSTM_block_dim, batch_first=True)
        self.MLP_unemp_1 = nn.Linear(LSTM_block_dim, MLP_block_dim)
        self.MLP_unemp_2 = nn.Linear(MLP_block_dim, MLP_block_dim)

        self.output_unemp = nn.Linear(MLP_block_dim, 1)

        self.LSTM_block_inf = nn.LSTM(
            LSTM_block_dim, LSTM_block_dim, batch_first=True)
        self.MLP_inf_1 = nn.Linear(LSTM_block_dim, MLP_block_dim)
        self.MLP_inf_2 = nn.Linear(MLP_block_dim, MLP_block_dim)

        self.output_inf = nn.Linear(MLP_block_dim, 1)

    def forward(self, x):
        # feed through LSTM blocks
        x, _ = self.LSTM_block_1(x)
        x, _ = self.LSTM_block_2(x)
        x, _ = self.LSTM_block_3(x)

        # unemp head
        x_unemp, _ = self.LSTM_block_unemp(x)
        x_unemp = self.batch_norm(x_unemp[:, -1, :])
        x_unemp = torch.relu(self.MLP_unemp_1(x_unemp))
        x_unemp = self.dropout(x_unemp)
        x_unemp = torch.relu(self.MLP_unemp_2(x_unemp))
        unemp_out = self.output_unemp(x_unemp)

        # inf head
        x_inf, _ = self.LSTM_block_inf(x)
        x_inf = self.batch_norm(x_inf[:, -1, :])
        x_inf = torch.relu(self.MLP_inf_1(x_inf))
        x_inf = self.dropout(x_inf)
        x_inf = torch.relu(self.MLP_inf_2(x_inf))
        inf_out = self.output_inf(x_inf)

        return unemp_out, inf_out

# training timeeeeee


def train_model(model, train_loader, validation_loader, loss_function, optimiser, epochs, patience=5):

    best_validation_loss = 1_000
    patience_num = 0
    patience_treshold = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, z_batch in train_loader:
            # was getting a bug sometimes running, seems that some batches had size 1
            if X_batch.size(0) == 1:
                continue
            optimiser.zero_grad()
            y_pred, z_pred = model(X_batch)
            loss_y = loss_function(y_pred.squeeze(), y_batch)
            loss_z = loss_function(z_pred.squeeze(), z_batch)
            loss = loss_y + loss_z
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # compute validation loss at each epoch
        validation = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch, z_batch in validation_loader:
                y_pred, z_pred = model(X_batch)
                loss_y = loss_function(y_pred.squeeze(), y_batch)
                loss_z = loss_function(z_pred.squeeze(), z_batch)
                loss = loss_y + loss_z
                validation += loss.item()

        validation_loss = validation / len(validation_loader)
        # print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}")

        # rudimentary implementation of early stopping, it includes a burn-in phase, not sure if good idea or not
        if validation_loss < best_validation_loss or epoch < 20:
            best_validation_loss = validation_loss
            patience_num = 0
        else:
            patience_num += 1
            if patience_num >= patience_treshold:
                break

    return train_loss

# testing timeeeeee


def test_model(model, test_loader):
    model.eval()  # included to make sure dropout doesn't apply during testing
    preds_y, preds_z, actuals_y, actuals_z = [], [], [], []
    for X_batch, y_batch, z_batch in test_loader:
        y_pred, z_pred = model(X_batch)
        preds_y.extend(y_pred.squeeze().tolist())
        preds_z.extend(z_pred.squeeze().tolist())
        actuals_y.extend(y_batch.tolist())
        actuals_z.extend(z_batch.tolist())

    # illustrate how the model does on the test set
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(actuals_y, label="Actual y")
    plt.plot(preds_y, label="Predicted y")
    plt.legend()
    plt.title("Predictions vs Actual Values for y")
    plt.xlabel("Window Index")
    plt.ylabel("Value")

    plt.subplot(2, 1, 2)
    plt.plot(actuals_z, label="Actual z")
    plt.plot(preds_z, label="Predicted z")
    plt.legend()
    plt.title("Predictions vs Actual Values for z")
    plt.xlabel("Window Index")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

# extract windows of the form (x_t, ..., x_{t+h}, y_{t+h+1}) to form train, test and val


class Windowify(Dataset):
    def __init__(self, data, time_steps):
        self.X, self.y, self.z = self.windowify(data, time_steps)

    def windowify(self, data, time_steps):
        X, y, z = [], [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i: i + time_steps, : -2].values)
            y.append(data.iloc[i + time_steps, -2])
            z.append(data.iloc[i + time_steps, -1])

        # the console whines if X and y aren't converted into float32s
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(z, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]


def hyperparam_search(param_grid, data, covariate_dim, epochs, number_of_trials):

    best_loss = 100
    best_hyperparams = None
    for _ in range(number_of_trials):

        # randomly sample combination of hyperparameters from pre-made grid
        hyperparams = {key: random.choice(value)
                       for key, value in param_grid.items()}
        print("Sampled hyperparameters:", hyperparams)

        # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
        dataset = Windowify(data, hyperparams['time_steps'])

        training_len = int(0.6 * len(dataset))
        vallidation_len = int(0.2 * len(dataset))

        # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
        # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
        train_dataset = torch.utils.data.Subset(
            dataset, list(range(training_len)))
        validation_dataset = torch.utils.data.Subset(dataset, list(
            range(training_len, training_len + vallidation_len)))

        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        validation_loader = DataLoader(
            validation_dataset, batch_size=hyperparams['batch_size'])

        # train model on sampled hyperparams and see how it does
        model = LSTM_model(
            covariate_dim, hyperparams['LSTM_block_dim'], hyperparams['MLP_block_dim'])
        optimiser = optim.Adam(
            model.parameters(), lr=hyperparams['learning_rate'])
        loss_function = nn.MSELoss()
        validation_loss = train_model(
            model, train_loader, validation_loader, loss_function, optimiser, epochs)
        print(f"Validation loss: {validation_loss:.4f}\n")

        # store whichever parameters minimise the validation loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_hyperparams = hyperparams
    print('\nBest loss:', best_loss)

    return best_hyperparams


if __name__ == "__main__":

    # choose target_covariate (UNRATE or CPIAUCSL) and load data
    data = pd.read_csv(
        "/Users/meesvandartel/Desktop/Coursework/Deep Learning/coding/deep-learning-project-dewi/data/monthly_filled_both.csv")

    # ========== 1) hyperparameter search ==========#

    # to be randomly sampled from in search of best hyper parameter combination
    hyperparam_grid_dict = {
        "batch_size": [24, 32, 42],
        "learning_rate": [0.0003, 0.0007],
        "LSTM_block_dim": [32, 64, 96],
        "MLP_block_dim": [8, 16, 32],
        "time_steps": [1, 3, 6]
    }

    # perform search for best hyperparam combination
    covariate_dim = data.shape[1] - 2  # final column is target covariate
    epochs = 1000  # early stopping is used so this value is kind of an afterthought
    number_of_trials = 10
    best_hyperparams = hyperparam_search(
        hyperparam_grid_dict, data, covariate_dim, epochs, number_of_trials)
    print(best_hyperparams)

    # ========== 2) test 'best' hyperparameters from search ==========#

    # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
    dataset = Windowify(data, best_hyperparams['time_steps'])

    training_len = int(0.6 * len(dataset))
    vallidation_len = int(0.2 * len(dataset))

    # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
    # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
    train_dataset = torch.utils.data.Subset(dataset, list(range(training_len)))
    validation_dataset = torch.utils.data.Subset(dataset, list(
        range(training_len, training_len + vallidation_len)))
    test_dataset = torch.utils.data.Subset(dataset, list(
        range(training_len + vallidation_len, len(dataset))))  # last 20%

    train_loader = DataLoader(
        train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=best_hyperparams['batch_size'])
    test_loader = DataLoader(
        test_dataset, batch_size=best_hyperparams['batch_size'])

    # test model on best hyperparams and see how it does
    model = MultiTask_model(
        covariate_dim, best_hyperparams['LSTM_block_dim'], best_hyperparams['MLP_block_dim'])
    optimiser = optim.Adam(
        model.parameters(), lr=best_hyperparams['learning_rate'])
    loss_function = nn.MSELoss()

    train_model(model, train_loader, validation_loader,
                loss_function, optimiser, epochs)
    test_model(model, test_loader)
