import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define MLP sub-model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, dropout_prob=0.2):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))  # Added Dropout
            in_features = neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define full model with three sub-MLPs
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_sizes, hidden_layers, neurons_per_layer, dropout_prob):
        super(NeuralNetworkModel, self).__init__()
        self.mlp1 = MLP(input_sizes[0], hidden_layers, neurons_per_layer, dropout_prob=dropout_prob)
        self.mlp2 = MLP(input_sizes[1], hidden_layers, neurons_per_layer, dropout_prob=dropout_prob)
        self.mlp3 = MLP(input_sizes[2], hidden_layers, neurons_per_layer, dropout_prob=dropout_prob)
    
    def forward(self, x1, x2, x3):
        out1 = self.mlp1(x1)
        out2 = self.mlp2(x2)
        out3 = self.mlp3(x3)
        final_out = out1 + out2 + out3
        return final_out

# Training timeee
def train_model(model, train_loader, val_loader, loss_function, optimiser, epochs, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1_batch, x2_batch, x3_batch, y_batch in train_loader:
            optimiser.zero_grad()
            outputs = model(x1_batch, x2_batch, x3_batch)
            loss = loss_function(outputs.squeeze(), y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x1_batch, x2_batch, x3_batch, y_batch in val_loader:
                outputs = model(x1_batch, x2_batch, x3_batch)
                loss = loss_function(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    
    # Plot train & val curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/HNN_loss_curve.pdf")  # Save figure
    plt.show()

# Testing 
def test_model(model, test_loader, purpose):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, y_batch in test_loader:
            outputs = model(x1_batch, x2_batch, x3_batch).squeeze()
            predictions.extend(outputs.tolist())
            actuals.extend(y_batch.tolist())
    test_loss = mean_squared_error(actuals, predictions)
    plt.figure()
    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title(f'Predictions vs Actual Values, {purpose} Loss: {np.round(test_loss, 8)}')
    plt.xlabel('Window Index')
    plt.ylabel('Inflation')
    plt.savefig(f'figures/{purpose}_NN_inflation.pdf')
    print(f'{purpose} loss:', np.round(test_loss, 8))


# Split data into train and test
def preprocess_data(data, scale_y = True, scale_x = True):
  X = data.iloc[:, 1:].to_numpy()
  y = data['y'].to_numpy()

  # number of samples
  num_rows = data.shape[0]
  train_val_split_id = int(num_rows * 0.6)
  val_test_split_id = int(num_rows * 0.8)

  # Train test split 
  X_train = X[:train_val_split_id, :]
  y_train = y[:train_val_split_id]
  X_val = X[train_val_split_id:val_test_split_id, :]
  y_val = y[train_val_split_id:val_test_split_id]
  X_test = X[val_test_split_id:, :]
  y_test = y[val_test_split_id:]

  y_mean = y_train.mean()
  y_std = y_train.std()

  # Standardize the variables
  if scale_x == True:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) 
    X_test = scaler.transform(X_test)

  if scale_y == True:
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

  # Convert to tensor
  X_train = torch.tensor(X_train, dtype = torch.float)
  X_val = torch.tensor(X_val, dtype=torch.float)
  X_test = torch.tensor(X_test, dtype = torch.float)

  y_train = torch.tensor(y_train, dtype = torch.float)
  y_val = torch.tensor(y_val, dtype=torch.float)
  y_test = torch.tensor(y_test, dtype = torch.float)

  return X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std


# Function that append two list of string
def append_list(list1, list2):
    list3 = []
    for i in list1:
        for j in list2:
            list3.append(i+j)
    return list3

# Function that return position of columns
def find_position(col_names, string_col):
    pos = []
    for i in string_col:
      if i in col_names:
        pos.append(col_names.index(i))
      else:
        print(f"Column '{i}' not found in col_names.")
    return pos

def get_hemishere_data(col_names):
    # Which lags to use ? (all the first part of the column name in the data)
    first_part = ["L0_", "L1_", "L2_", "L3_","L1_MARX_", "L3_MARX_", "L7_MARX_"]

    # Hemisphere 1 : Real activity
    labor_name = ["PAYEMS", "USPRIV", "MANEMP", "SRVPRD", "USGOOD", "DMANEMP", "NDMANEMP", "USCONS", "USEHS",
                "USFIRE", "USINFO", "USPBS", "USLAH", "USSERV", "USMINE", "USTPU", "USGOVT", "USTRADE",
                "USWTRADE", "CES9091000001", "CES9092000001", "CES9093000001", "CE16OV", "CIVPART",
                "UNRATE", "UNRATESTx", "UNRATELTx", "LNS14000012", "LNS14000025", "LNS14000026",
                "UEMPLT5", "UEMP5TO14", "UEMP15T26", "UEMP27OV", "LNS13023621", "LNS13023557",
                "LNS13023705", "LNS13023569", "LNS12032194", "HOABS", "HOAMS", "HOANBS", "AWHMAN",
                "AWHNONAG", "AWOTMAN", "HWIx", "UEMPMEAN", "CES0600000007", "HWIURATIOx", "CLAIMSx",
                "GDPC1","PCECC96", "GPDIC1", "OUTNFB", "OUTBS", "OUTMS", "INDPRO", "IPFINAL", "IPCONGD", 
                "IPMAT", "IPDMAT", "IPNMAT", "IPDCONGD", "IPB51110SQ", "IPNCONGD", "IPBUSEQ", "IPB51220SQ",
                "TCU", "CUMFNS","IPMANSICS", "IPB51222S", "IPFUELS"]
    hemisphere_var = append_list(first_part, labor_name)
    x_pos1 = find_position(col_names, hemisphere_var)

    # Hemisphere 2 : Short-run expectations
    price_name = ["PCECTPI", "PCEPILFE", "GDPCTPI", "GPDICTPI", "IPDBS","CPILFESL", "CPIAPPSL","CPITRNSL",
                "CPIMEDSL", "CUSR0000SAC","CUSR0000SAD", "WPSFD49207","PPIACO", "WPSFD49502", "WPSFD4111",
                "PPIIDC", "WPSID61", "WPSID62","CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5",
                "CUSR0000SEHC","spf_cpih1", "spf_cpi_currentYrs", "inf_mich"]
    hemisphere_var = append_list(first_part, price_name)
    x_pos2 = find_position(col_names, hemisphere_var)
    x_pos2 = list(range(0,4)) + x_pos2

    # Hemisphere 3 : Commodities
    commodities_name = ["WPU0531","WPU0561", "OILPRICEx", "PPICMM"]
    hemisphere_var = append_list(first_part, commodities_name)
    x_pos3 = find_position(col_names, hemisphere_var)

    return x_pos1, x_pos2, x_pos3

# Load data
def load_data():
    # import data from csv file
    data = pd.read_csv('data/dataUS.csv')

    # dates vector
    dates = data.iloc[:, 0]
    dates = dates.astype(str)
    data = data.iloc[:, 1:]

    # get column names
    col_names = list(data.columns)[1:]

    # Standardize data
    X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = preprocess_data(data, scale_y = True, scale_x = True)
    X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

    
    return X_train, X_val, X_test, y_train, y_val, y_test, col_names

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, col_names = load_data()

    x_pos1, x_pos2, x_pos3 = get_hemishere_data(col_names)
    input_sizes = [len(x_pos1), len(x_pos2), len(x_pos3)] 
    hidden_layers = 5
    neurons_per_layer = 400
    learning_rate = 0.005
    epochs = 500
    dropout_prob = 0.2
    patience = 8

    # Extract columns based on x_pos1, x_pos2, x_pos3
    X1_train, X1_val, X1_test = X_train[:, x_pos1], X_val[:, x_pos1], X_test[:, x_pos1]
    X2_train, X2_val, X2_test = X_train[:, x_pos2], X_val[:, x_pos2], X_test[:, x_pos2]
    X3_train, X3_val, X3_test = X_train[:, x_pos3], X_val[:, x_pos3], X_test[:, x_pos3]

    # train
    train_data = torch.utils.data.TensorDataset(X1_train.clone().detach(
    ), X2_train.clone().detach(), X3_train.clone().detach(), y_train.clone().detach())

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=False)
    
    # val
    val_data = torch.utils.data.TensorDataset(X1_val.clone().detach(
    ), X2_val.clone().detach(), X3_val.clone().detach(), y_val.clone().detach())

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=32, shuffle=False)

    # test
    test_data = torch.utils.data.TensorDataset(X1_test.clone().detach(
    ), X2_test.clone().detach(), X3_test.clone().detach(), y_test.clone().detach())
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False)

    model = NeuralNetworkModel(input_sizes, hidden_layers, neurons_per_layer, dropout_prob=dropout_prob)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    train_model(model, train_loader, val_loader, loss_function, optimiser, epochs, patience=patience)
    test_model(model, test_loader, 'Test')
    test_model(model, train_loader, 'Train')