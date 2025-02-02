from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
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
        
        # unclear if useful but can't hurt
        self.batch_norm = nn.BatchNorm1d(LSTM_block_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.MLP_layer_1 = nn.Linear(LSTM_block_dim, MLP_block_dim)
        self.MLP_layer_2 = nn.Linear(MLP_block_dim, MLP_block_dim)
        
        self.output = nn.Linear(MLP_block_dim, 1)

    def forward(self, x):

        # feed through LSTM blocks
        x, h = self.LSTM_block_1(x)
        x, h = self.LSTM_block_2(x)
        x, h = self.LSTM_block_3(x)

        # unclear if useful but can't hurt
        x = self.batch_norm(x[:, -1, :])
        x = self.dropout(x)

        # feed output of final LSTM block as input to MLP
        x = torch.relu(self.MLP_layer_1(x))
        x = torch.relu(self.MLP_layer_2(x))
        x = self.output(x)

        return x

# training timeeeeee
def train_model(model, train_loader, validation_loader, loss_function, optimiser, epochs):

    best_validation_loss = 1_000
    patience_num = 0
    patience_treshold = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            if X_batch.size(0) == 1: # was getting a bug sometimes running, seems that some batches had size 1
                continue
            optimiser.zero_grad()
            outputs = model(X_batch)
            loss = loss_function(outputs.squeeze(), y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # compute validation loss at each epoch
        validation = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                outputs = model(X_batch)
                loss = loss_function(outputs.squeeze(), y_batch)
                validation += loss.item()
        validation_loss = validation / len(validation_loader)
        # print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}")

        # rudimentary implementation of early stopping, it includes a burn-in phase, probably a good idea
        if validation_loss < best_validation_loss or epoch < 20:
            best_validation_loss = validation_loss
            patience_num = 0
        else:
            patience_num += 1
            if patience_num >= patience_treshold:
                break
        
    return best_validation_loss

# testing timeeeeee
def test_model(model, test_loader, target_covariate):
    
    loss_func = nn.MSELoss()
    test_loss = 0
    
    model.eval() # included to make sure dropout doesn't apply during testing
    preds, actuals = [], []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        preds.extend(outputs.tolist())
        actuals.extend(y_batch.tolist())
        test_loss += loss_func(outputs, y_batch).item()
    test_loss /= len(test_loader)

    # illustrate how the model does on the test set
    plt.figure()
    plt.plot(actuals, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title(f'Predictions vs Actual Values, test loss: {np.round(test_loss, 8)}')
    plt.xlabel('Window Index')
    plt.ylabel('Value')
    plt.savefig(f'figures/test_LSTM_single_pred_{target_covariate}.pdf')

    print('Test loss:', np.round(test_loss, 8))

# extract windows of the form (x_t, ..., x_{t+h}, y_{t+h+1}) to form train, test and val
class Windowify(Dataset):
    def __init__(self, data, time_steps):
        self.X, self.y = self.windowify(data, time_steps)

    def windowify(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i : i + time_steps, : -1].values)
            y.append(data.iloc[i + time_steps, -1])
        
        # the console whines if X and y aren't converted into float32s
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def hyperparam_search(param_grid, data, covariate_dim, epochs, number_of_trials):

    best_loss = 100
    best_hyperparams = None
    for _ in range(number_of_trials):
        
        # randomly sample combination of hyperparameters from pre-made grid
        hyperparams = {key : random.choice(value) for key, value in param_grid.items()}
        print("Sampled hyperparameters:", hyperparams)

        # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
        dataset = Windowify(data, hyperparams['time_steps'])

        training_len    = int(0.6 * len(dataset))
        vallidation_len = int(0.2 * len(dataset))

        # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
        # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
        train_dataset      = torch.utils.data.Subset(dataset, list(range(training_len)))
        validation_dataset = torch.utils.data.Subset(dataset, list(range(training_len, training_len + vallidation_len)))
        
        train_loader      = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=hyperparams['batch_size'])

        # train model on sampled hyperparams and see how it does 
        model = LSTM_model(covariate_dim, hyperparams['LSTM_block_dim'], hyperparams['MLP_block_dim'])
        optimiser = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        loss_function = nn.MSELoss()
        validation_loss = train_model(model, train_loader, validation_loader, loss_function, optimiser, epochs)
        print(f"Validation loss: {validation_loss:.4f}\n")

        # store whichever parameters minimise the validation loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_hyperparams = hyperparams
    print('Best validation loss:', best_loss)

    return best_hyperparams

if __name__ == "__main__":

    # the target covariates relevant to our task are unemployment (UNRATE) and inflation (CPIAUCSL)
    for target_covariate in ['UNRATE', 'CPIAUCSL']:
        
        #========== 0) load corresponding dataset ==========#
        data = pd.read_csv(f'data/monthly_filled_{target_covariate}.csv')

        #========== 1) hyperparameter search ==========#
        
        # to be randomly sampled from in search of best hyper parameter combination
        hyperparam_grid_dict = {
            "batch_size": [16, 32, 64],
            "learning_rate": [0.001, 0.01, 0.1],
            "LSTM_block_dim": [8, 16, 32, 64, 128],
            "MLP_block_dim": [4, 8, 16, 32, 64],
            "time_steps": [6, 10, 14, 18, 20, 24, 28, 32, 36]
        }

        # perform search for best hyperparam combination
        covariate_dim = data.shape[1] - 1 # final column is target covariate
        epochs = 100 # early stopping is used so this value is kind of an afterthought
        number_of_trials = 5
        best_hyperparams = hyperparam_search(hyperparam_grid_dict, data, covariate_dim, epochs, number_of_trials)
        print('Best hyperparameters:', best_hyperparams)

        #========== 2) test 'best' hyperparameters from search ==========#

        # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
        dataset = Windowify(data, best_hyperparams['time_steps'])

        training_len    = int(0.6 * len(dataset))
        vallidation_len = int(0.2 * len(dataset))

        # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
        # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
        train_dataset      = torch.utils.data.Subset(dataset, list(range(training_len)))
        validation_dataset = torch.utils.data.Subset(dataset, list(range(training_len, training_len + vallidation_len)))
        test_dataset       = torch.utils.data.Subset(dataset, list(range(training_len + vallidation_len, len(dataset)))) # last 20%
        
        train_loader      = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=best_hyperparams['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=best_hyperparams['batch_size'])

        # train + test model on best hyperparams and see how it does 
        model = LSTM_model(covariate_dim, best_hyperparams['LSTM_block_dim'], best_hyperparams['MLP_block_dim'])
        optimiser = optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
        loss_function = nn.MSELoss()
        
        train_model(model, train_loader, validation_loader, loss_function, optimiser, epochs)
        test_model(model, test_loader, target_covariate)
