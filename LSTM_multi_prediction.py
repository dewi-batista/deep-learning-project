from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

# architecture construction
class MultiLSTM_model(nn.Module):

    def __init__(self, covariate_dim, shared_layer_dim, separate_layer_dim, MLP_block_dim):
        super(MultiLSTM_model, self).__init__()
        
        self.shared_LSTM_1 = nn.LSTM(covariate_dim, shared_layer_dim, batch_first=True)
        self.shared_LSTM_2 = nn.LSTM(shared_layer_dim, shared_layer_dim, batch_first=True)
        self.shared_LSTM_3 = nn.LSTM(shared_layer_dim, shared_layer_dim, batch_first=True)
        
        self.cpi_LSTM = nn.LSTM(shared_layer_dim, separate_layer_dim, batch_first=True)
        self.inflation_dense_1 = nn.Linear(separate_layer_dim, MLP_block_dim)
        self.inflation_dense_2 = nn.Linear(MLP_block_dim, MLP_block_dim)
        self.inflation_out = nn.Linear(MLP_block_dim, 1)
        
        self.unemp_LSTM = nn.LSTM(shared_layer_dim, separate_layer_dim, batch_first=True)
        self.unemp_dense_1 = nn.Linear(separate_layer_dim, MLP_block_dim)
        self.unemp_dense_2 = nn.Linear(MLP_block_dim, MLP_block_dim)
        self.output_unemployment = nn.Linear(MLP_block_dim, 1)

    def forward(self, x):
        
        # feed through shared LSTM blocks
        x, _ = self.shared_LSTM_1(x)
        x, _ = self.shared_LSTM_2(x)
        x, _ = self.shared_LSTM_3(x)

        # inflation branch
        output_inflation, _ = self.cpi_LSTM(x)
        output_inflation = output_inflation[:, -1, :]
        output_inflation = torch.relu(self.inflation_dense_1(output_inflation))
        output_inflation = torch.relu(self.inflation_dense_2(output_inflation))
        output_inflation = self.inflation_out(output_inflation)

        # unemployment branch
        output_unemployment, _ = self.unemp_LSTM(x)
        output_unemployment = output_unemployment[:, -1, :]
        output_unemployment = torch.relu(self.unemp_dense_1(output_unemployment))
        output_unemployment = torch.relu(self.unemp_dense_2(output_unemployment))
        output_unemployment = self.output_unemployment(output_unemployment)

        return torch.cat([output_inflation, output_unemployment], dim=1)

# training timeeeeee
def train_model(model, train_loader, validation_loader, loss_function, optimiser, epochs):
    best_validation_loss = 1_000
    patience_num = 0
    patience_threshold = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            if X_batch.size(0) == 1: # was getting a bug sometimes running, seems that some batches had size 1
                continue
            optimiser.zero_grad()
            outputs = model(X_batch)

            # compute MSEs separately and sum
            inflation_mse = nn.functional.mse_loss(outputs[:,0], y_batch[:,0])
            unemp_mse = nn.functional.mse_loss(outputs[:,1], y_batch[:,1])
            loss = inflation_mse + unemp_mse

            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        
        # compute validation loss at each epoch
        validation = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                outputs = model(X_batch)
                inflation_mse = nn.functional.mse_loss(outputs[:,0], y_batch[:,0])
                unemp_mse = nn.functional.mse_loss(outputs[:,1], y_batch[:,1])
                loss = inflation_mse + unemp_mse
                validation += loss.item()
        validation_loss = validation / len(validation_loader)
        # print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}")
        
        # rudimentary implementation of early stopping, it includes a burn-in phase, probably a good idea
        if validation_loss < best_validation_loss or epoch < 20:
            best_validation_loss = validation_loss
            patience_num = 0
        else:
            patience_num += 1
            if patience_num >= patience_threshold:
                break
    return best_validation_loss

# testing timeeeeee
def test_model(model, test_loader):

    test_loss = 0
    
    model.eval() # included to make sure dropout doesn't apply during testing
    preds_inflation, preds_unemployment = [], []
    actuals_inflation, actuals_unemployment = [], []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        
        inf_mse = nn.functional.mse_loss(outputs[:,0], y_batch[:,0])
        unemp_mse = nn.functional.mse_loss(outputs[:,1], y_batch[:,1])
        loss = inf_mse + unemp_mse
        test_loss += loss.item()
        
        pred_inf = outputs[:, 0].tolist()
        pred_unemp = outputs[:, 1].tolist()
        
        act_inf = y_batch[:, 0].tolist()
        act_unemp = y_batch[:, 1].tolist()
        
        preds_inflation.extend(pred_inf)
        preds_unemployment.extend(pred_unemp)
        
        actuals_inflation.extend(act_inf)
        actuals_unemployment.extend(act_unemp)
    test_loss /= len(test_loader)

    # illustrate how the model does on the inflation test set
    plt.figure()
    plt.plot(actuals_inflation, label='Actual Infl.')
    plt.plot(preds_inflation, label='Pred. Infl.')
    plt.legend()
    plt.title(f'Inflation Forecast (MSE: {test_loss:.3f})')
    plt.savefig('figures/test_LSTM_multi_pred_inflation.pdf')

    # illustrate how the model does on the unemployment test set
    plt.figure()
    plt.plot(actuals_unemployment, label='Actual Unemp.')
    plt.plot(preds_unemployment, label='Pred. Unemp.')
    plt.legend()
    plt.title(f'Unemployment Forecast (MSE: {test_loss:.3f})')
    plt.savefig('figures/test_LSTM_multi_pred_unemp.pdf')
    
    print('Test loss:', round(test_loss, 3))

# extract windows of the form (x_t, ..., x_{t+h}, y_{1,t+h+1}, y_{2,t+h+1}) to form train, test and val
class Windowify(Dataset):
    def __init__(self, data, time_steps):
        self.X, self.y = self.windowify(data, time_steps)

    def windowify(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i : i + time_steps, :-2].values)
            y.append(data.iloc[i + time_steps, -2:].values)
        
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
        hyperparams = {k: random.choice(v) for k,v in param_grid.items()}
        print('Sampled hyperparams:', hyperparams)

        # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
        dataset = Windowify(data, hyperparams['time_steps'])
        training_len = int(0.6 * len(dataset))
        vallidation_len = int(0.2 * len(dataset))

        # at first, the window extractions were randomly assigned to train, validation and test but that ofcourse
        # leads to potential overfitting (since windows heavily overlap), now they are split chronologically
        train_dataset      = torch.utils.data.Subset(dataset, list(range(training_len)))
        validation_dataset = torch.utils.data.Subset(dataset, list(range(training_len, training_len + vallidation_len)))

        train_loader      = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=hyperparams['batch_size'])

        # train model on sampled hyperparams and see how it does 
        model = MultiLSTM_model(covariate_dim, hyperparams['shared_layer_dim'], hyperparams['separate_layer_dim'], hyperparams['MLP_block_dim'])
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

if __name__ == '__main__':

    #========== 0) load corresponding dataset ==========#
    data = pd.read_csv('data/monthly_filled_multi.csv')

    #========== 1) hyperparameter search ==========#

    # to be randomly sampled from in search of best hyper parameter combination
    hyperparam_grid_dict = {
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.1],
        'time_steps': [6, 10, 14, 18, 20, 24, 28, 32, 36],
        'shared_layer_dim': [8, 16, 32, 64, 128],
        'separate_layer_dim': [8, 16, 32, 64, 128],
        'MLP_block_dim': [4, 8, 16, 32, 64],
    }

    # perform search for best hyperparam combination
    covariate_dim = data.shape[1] - 2 # final two columns are target covariates
    epochs = 100 # early stopping is used so this value is kind of an afterthought
    number_of_trials = 10
    best_hyperparams = hyperparam_search(hyperparam_grid_dict, data, covariate_dim, epochs, number_of_trials)
    print('Best hyperparameters:', best_hyperparams)

    #========== 2) test 'best' hyperparameters from search ==========#

    # windowify and assign 60%, 20% and 20% for training, validation and testing, repectively
    dataset = Windowify(data, best_hyperparams['time_steps'])

    training_len    = int(0.6 * len(dataset))
    vallidation_len = int(0.2 * len(dataset))

    train_dataset      = torch.utils.data.Subset(dataset, list(range(training_len)))
    validation_dataset = torch.utils.data.Subset(dataset, list(range(training_len, training_len + vallidation_len)))
    test_dataset       = torch.utils.data.Subset(dataset, list(range(training_len + vallidation_len, len(dataset))))

    train_loader      = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=best_hyperparams['batch_size'])
    test_loader       = DataLoader(test_dataset, batch_size=best_hyperparams['batch_size'])

    # train + test model on best hyperparams and see how it does 
    model = MultiLSTM_model(covariate_dim, best_hyperparams['shared_layer_dim'], best_hyperparams['separate_layer_dim'], best_hyperparams['MLP_block_dim'])
    optimiser = optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
    criterion = nn.MSELoss()

    train_model(model, train_loader, validation_loader, criterion, optimiser, epochs)
    test_model(model, test_loader)
