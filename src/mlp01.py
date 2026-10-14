import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from test01 import load_data, adding_columns, create_train_test_split, data_cleaning

IMAGES_DIR = "/home/rianr/pypro/myvenv/virtual-well-test/Images"
OUTPUTS_DIR = "/home/rianr/pypro/myvenv/virtual-well-test/outputs"

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32, 16), dropout=0.1):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def prepare_data(file_path):
    # same preprocessing as test01.main, without the exploratory plots
    data = load_data(file_path)
    data = adding_columns(data)
    X_train, X_test, y_train, y_test = create_train_test_split(data)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data, train_medians = data_cleaning(train_data)
    X_train = train_data.drop(columns=['GROSS_FLUID'])
    y_train = train_data['GROSS_FLUID']
    test_data = test_data.dropna()
    X_test = test_data.drop(columns=['GROSS_FLUID'])
    y_test = test_data['GROSS_FLUID']
    #WATER_CUT and GAS_LIQUID_RATIO are computed from GROSS_FLUID (the target), drop them to avoid leakage
    leak_cols = ['TEST_DATE', 'WATER_CUT', 'GAS_LIQUID_RATIO', 'FM_GAS']
    to_drop = ['DURATION']
    X_train = X_train.drop(columns=leak_cols + to_drop)
    X_test = X_test.drop(columns=leak_cols + to_drop)
    return X_train, X_test, y_train, y_test


def to_tensor(arr):
    return torch.tensor(np.asarray(arr), dtype=torch.float32, device=device)


def train_mlp(X_train, y_train, epochs=500, batch_size=64, lr=1e-3, patience=30):
    # hold out part of the training set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # MLPs need scaled inputs; scale the target too so the loss is well conditioned
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.values.reshape(-1, 1))
    X_tr_t = to_tensor(x_scaler.transform(X_tr))
    X_val_t = to_tensor(x_scaler.transform(X_val))
    y_tr_t = to_tensor(y_scaler.transform(y_tr.values.reshape(-1, 1)).ravel())
    y_val_t = to_tensor(y_scaler.transform(y_val.values.reshape(-1, 1)).ravel())

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    model = MLP(X_tr_t.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    history = {'train': [], 'val': []}
    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, x_scaler, y_scaler, history


def predict(model, x_scaler, y_scaler, X):
    model.eval()
    with torch.no_grad():
        y_scaled = model(to_tensor(x_scaler.transform(X))).cpu().numpy()
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def plot_results(y_test, y_pred, r2, history):
    #loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss (scaled)')
    plt.yscale('log')
    plt.title('MLP training history')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/mlp_loss.png")
    plt.close()
    #actual vs predicted, R^2 in the title
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=15)
    plt.plot(lims, lims, 'r--', label='Perfect prediction')
    plt.xlabel('Actual GROSS_FLUID')
    plt.ylabel('Predicted GROSS_FLUID')
    plt.title(f'MLP (R^2 = {r2:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/mlp_r2_score.png")
    plt.close()


def main(file_path):
    X_train, X_test, y_train, y_test = prepare_data(file_path)
    print('Features: ', list(X_train.columns))
    print('Train size: ', len(X_train), ' Test size: ', len(X_test))
    model, x_scaler, y_scaler, history = train_mlp(X_train, y_train)
    y_pred = predict(model, x_scaler, y_scaler, X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MLP - Mean Squared Error: {mse}")
    print(f"MLP - R^2 Score: {r2}")
    plot_results(y_test, y_pred, r2, history)
    torch.save({'model_state': model.state_dict(),
                'features': list(X_train.columns)}, f"{OUTPUTS_DIR}/mlp_model.pt")


if __name__ == "__main__":
    main("/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv")
