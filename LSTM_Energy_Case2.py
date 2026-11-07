import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تعریف مدل LSTM
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, output_size=1, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions, (hn, cn)

# تابع Sliding Window
def sliding_windows(data, seq_length=24):
    x, y = [], []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

# کلاس EarlyStopping
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ارزیابی
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def safe_mape(y_true, y_pred, threshold=1e-6):
    diff = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), threshold)
    return np.mean(diff / denom) * 100

# پیش‌بینی batched
def predict_batched(model, x_data, y_data, sc, batch_size=256):
    model.eval()
    dataset = DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size)
    preds, targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            output, _ = model(batch_x)
            preds.append(output.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    preds = sc.inverse_transform(np.concatenate(preds))
    targets = sc.inverse_transform(np.concatenate(targets))
    return preds, targets

# آموزش مدل با Train/Validation
def train_with_validation(num_epochs, model, train_loader, val_loader, criterion, optimizer):
    early_stopping = EarlyStopping(patience=20, min_delta=1e-2)
    best_val_loss = np.inf
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # EarlyStopping براساس validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}, Val Loss: {val_loss:.6f}")
            break

    model.load_state_dict(best_model_state)
    return train_losses, val_losses

# ==================================================================
print("Data Energy")

# بارگذاری داده
train_df = pd.read_csv('data/train/energy.csv')[["MW"]]
test_df = pd.read_csv('data/test/energy_t.csv')[["MW"]]

sc = MinMaxScaler()
train_data = sc.fit_transform(train_df)
test_data = sc.transform(test_df)

# ساخت دیتاهای sliding
seq_length = 24
x_train_full_np, y_train_full_np = sliding_windows(train_data, seq_length)
x_test_np, y_test_np = sliding_windows(test_data, seq_length)

# تقسیم به train/validation
x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(x_train_full_np, y_train_full_np, test_size=0.2 ,shuffle=True, random_state=42)

# تبدیل به Tensor
x_train = torch.Tensor(x_train_np).to(device)
y_train = torch.Tensor(y_train_np).to(device)
x_val = torch.Tensor(x_val_np).to(device)
y_val = torch.Tensor(y_val_np).to(device)
x_test = torch.Tensor(x_test_np).to(device)
y_test = torch.Tensor(y_test_np).to(device)

# DataLoader
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)

# تعریف مدل
model = LSTM(input_size=1, hidden_layer_size=64).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
train_losses, val_losses = train_with_validation(1500, model, train_loader, val_loader, criterion, optimizer)
torch.save(model, "lstm/lstm_energy.pt")

# رسم نمودار Train و Validation Loss
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Training Loss", color='green')
plt.plot(val_losses, label="Validation Loss", color='orange')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data_main/training_validation_loss_plot.png")
plt.show()

# پیش‌بینی روی train/test
predict_train, actual_train = predict_batched(model, x_train, y_train, sc)
predict_test, actual_test = predict_batched(model, x_test, y_test, sc)

# ذخیره پیش‌بینی‌ها
pd.DataFrame(predict_train).to_csv('data_main/predict_energy_train.csv', index=False)
pd.DataFrame(predict_test).to_csv('data_main/predict_energy_test.csv', index=False)

# انتقال به data_out
pd.DataFrame(predict_train, columns=["0"]).to_csv('data_out/energy_train.csv', mode='a', index=False, header=False)
pd.DataFrame(predict_test, columns=["0"]).to_csv('data_out/energy_test.csv', mode='a', index=False, header=False)

# ارزیابی نهایی
r2 = r2_score(actual_test, predict_test)
mape_value = mape(actual_test, predict_test)
safe_mape_value = safe_mape(actual_test, predict_test)
accuracy = 100 - mape_value
accuracy_safe = 100 - safe_mape_value
mae = mean_absolute_error(actual_test, predict_test)
rmse = math.sqrt(mean_squared_error(actual_test, predict_test))

print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape_value:.2f}%")
print(f"Safe_MAPE: {safe_mape_value:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Accuracy (Safe MAPE): {accuracy_safe:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# نمودار مقایسه واقعی و پیش‌بینی تست
plt.figure(figsize=(10, 5))
plt.plot(actual_test, color="red", label="Actual Energy (Test)")
plt.plot(predict_test, color="blue", label="Predicted Energy (Test)")
plt.title("Energy Prediction on Test Data")
plt.xlabel("Time Step")
plt.ylabel("Energy (MW)")
plt.legend()
plt.tight_layout()
plt.savefig("data_main/test_prediction_plot.png")
plt.show()
