# 匯入套件
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
import yfinance as yf

# ========== 全域變數：用於標準化與還原股價 ==========
close_mean = 0.0
close_std = 1.0

# 下載股價資料
ticker = "NVDA"
stock = yf.Ticker(ticker)
hist = stock.history(period="1y")
print(hist.head())
print(hist.tail())
hist.to_csv(f"{ticker}_historical_data.csv")

# 下載 NASDAQ 指數資料（可選）
nasdaq = yf.Ticker("^IXIC")
hist = nasdaq.history(period="1y")
print(hist.head())
hist.to_csv("NASDAQ_data.csv")

# 工具函式
def readData(f):
    return np.genfromtxt(f, delimiter=',', dtype=str)[1:]

def saveModel(net, path):
    torch.save(net.state_dict(), path)

# 自定義 Dataset 類別
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='gpu'):
        self.data = data.to(device)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

# 前處理函式（多維：收盤價 + 成交量）
def preprocess(data, flip=True):
    global close_mean, close_std
    close = np.array([row[1] for row in data]).astype(np.float64)
    volume = np.array([row[5] for row in data]).astype(np.float64)
    if flip:
        close = np.flip(close)
        volume = np.flip(volume)
    close_mean = close.mean()
    close_std = close.std()
    close = (close - close_mean) / close_std
    features = np.stack([close, volume], axis=1)
    return features

def inverse_transform(predicted_normalized):
    return [(x * close_std) + close_mean for x in predicted_normalized]

def train_test_split(data, percentage=0.8):
    train_size = int(len(data) * percentage)
    return data[:train_size], data[train_size:]

def transform_dataset(dataset, look_back=5, forecast_horizon=1):
    dataX = [dataset[i:(i + look_back), :] for i in range(len(dataset) - look_back - forecast_horizon + 1)]
    dataY = [dataset[i + look_back:i + look_back + forecast_horizon, 0] for i in range(len(dataset) - look_back - forecast_horizon + 1)]
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print("===== Sample from transform_dataset() =====")
    print("Sample X[0] shape:", dataX[0].shape)
    print("Sample X[0] values:\n", dataX[0])
    print("Sample Y[0]:", dataY[0])
    print("===============================\n")
    return torch.tensor(dataX, dtype=torch.float32), torch.tensor(dataY, dtype=torch.float32)

# 改良版 LSTM 模型（增加層級與正則化）
class LSTMPredictor(nn.Module):
    def __init__(self, feature_dim, output_len=1, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size=feature_dim, hidden_size=64, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = 64 * (2 if bidirectional else 1)
        self.model = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_len)
        )

    def predict(self, input):
        with torch.no_grad():
            raw = self.forward(input).squeeze().tolist()
            return inverse_transform(raw)

    def forward(self, input):
        r_out, _ = self.rnn(input)
        last_hidden = r_out[:, -1, :]
        return self.model(last_hidden)

# 訓練函式（加入 Early Stopping）
def trainer(net, criterion, optimizer, trainloader, devloader, epoch_n=100, path="./checkpoint/save.pt"):
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epoch_n):
        net.train()
        train_loss, valid_loss = 0.0, 0.0
        for inputs, labels, _ in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels.cpu())
            train_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()

        net.eval()
        for inputs, labels, _ in devloader:
            outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels.cpu())
            valid_loss += loss.item() * inputs.size(0)

        train_loss /= len(trainloader.dataset)
        valid_loss /= len(devloader.dataset)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f}")

        # Early Stopping 檢查
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience_counter = 0
            saveModel(net, path)  # 儲存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Finished Training")

# 測試函式
def tester(net, criterion, testloader):
    loss = 0
    with torch.no_grad():
        for inputs, labels, _ in testloader:
            outputs = net(inputs)
            loss += criterion(outputs.cpu(), labels.cpu())
    return loss.item()

# 取得訓練裝置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 處理資料
data = readData("./data/us_stock/NVDA.csv")
print("Num of samples:", len(data))
features = preprocess(data)
train, test = train_test_split(features)
look_back = 5
forecast_horizon = 5
feature_dim = features.shape[1]
trainX, trainY = transform_dataset(train, look_back, forecast_horizon)
testX, testY = transform_dataset(test, look_back, forecast_horizon)
trainset = Dataset(trainX, trainY, device)
testset = Dataset(testX, testY, device)
batch_size = 200
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# 建立模型
net = LSTMPredictor(feature_dim, output_len=forecast_horizon)
net.to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 開始訓練或載入模型
checkpoint = "checkpoint/save.pt"
if not os.path.isfile(checkpoint):
    os.makedirs("./checkpoint", exist_ok=True)
    trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=300, path=checkpoint)
else:
    net.load_state_dict(torch.load(checkpoint))

# 測試模型
test_loss = tester(net, criterion, testloader)
print("Test Result:", test_loss)

# 單筆預測範例（兩個 feature：收盤價 + 成交量）
sample_input = torch.tensor([[[121.66999816894531, 277593500], [119.52999877929688, 255501500], [115.43000030517578, 299686900], [117.5199966430664, 273426200], [118.52999877929688, 248250400]]], dtype=torch.float32).to(device)
predict = net.predict(sample_input)
print("Predicted Next 5 Days:", predict)
