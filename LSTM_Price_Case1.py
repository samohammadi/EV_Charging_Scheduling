import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-4):
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

# LSTM مدل
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
# Sliding window
def sliding_windows(data, seq_length=24):
    x, y = [], []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

print("Data Price")

# آموزش با ذخیره loss
def train(num_epochs, model, train_loader, criterion, optimizer):
    model.train()
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
    loss_list = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_loss)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_loss*100:.2f}%")

        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}, Loss: {avg_loss:.6f}")
            break

    return loss_list

# پیش‌بینی
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

# معیارها
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def safe_mape(y_true, y_pred, threshold=1e-6):
    diff = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), threshold)
    return np.mean(diff / denom) * 100

# بارگذاری داده
train_df = pd.read_csv('data/train/price.csv')[["PRICE"]]
test_df = pd.read_csv('data/test/price_t.csv')[["PRICE"]]

sc = MinMaxScaler()
train_data = sc.fit_transform(train_df)
test_data = sc.transform(test_df)

# ساخت دیتاست
seq_length = 24
x_train_np, y_train_np = sliding_windows(train_data, seq_length)
x_test_np, y_test_np = sliding_windows(test_data, seq_length)

x_train = torch.Tensor(x_train_np).to(device)
y_train = torch.Tensor(y_train_np).to(device)
x_test = torch.Tensor(x_test_np).to(device)
y_test = torch.Tensor(y_test_np).to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# تعریف مدل
model = LSTM(input_size=1, hidden_layer_size=64).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش و ذخیره loss
losses = train(num_epochs=1500, model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
torch.save(model, "lstm/lstm_price.pt")

# رسم نمودار loss
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss', color='purple')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data_main/training_loss_price.png")
plt.show()

# پیش‌بینی و ذخیره خروجی
predict_train, actual_train = predict_batched(model, x_train, y_train, sc)
predict_test, actual_test = predict_batched(model, x_test, y_test, sc)

pd.DataFrame(predict_train).to_csv('data_main/predict_price_train.csv', index=False)
pd.DataFrame(predict_test).to_csv('data_main/predict_price_test.csv', index=False)

# انتقال به فایل نهایی
training_raw5 = pd.read_csv('data_main/predict_price_train.csv')
dff5 = pd.DataFrame(training_raw5[["0"]])
dff5.to_csv('data_out/price_train.csv', mode='a', index=False, header=False)

training_raw5 = pd.read_csv('data_main/predict_price_test.csv')
dff5 = pd.DataFrame(training_raw5[["0"]])
dff5.to_csv('data_out/price_test.csv', mode='a', index=False, header=False)

# محاسبه معیارها
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

# نمودار پیش‌بینی تست
plt.figure(figsize=(10, 5))
plt.plot(actual_test, color="red", label="Actual Price (Test)")
plt.plot(predict_test, color="blue", label="Predicted Price (Test)")
plt.title("Price Prediction on Test Data")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("data_main/test_prediction_price.png")
plt.show()
