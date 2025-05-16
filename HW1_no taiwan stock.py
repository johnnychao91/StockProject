# 匯入套件
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解決 OpenMP 錯誤
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
import yfinance as yf
# 繪圖比較預測與真實值
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn

# ========== 全域變數：用於標準化與還原股價 ==========
close_mean = 0.0
close_std = 1.0

"""
# 下載股價資料
ticker = "NVDA"
stock = yf.Ticker(ticker)
hist = stock.history(period="1y")
print(hist.head())
print(hist.tail())
hist.to_csv(f"{ticker}_historical_data.csv")
"""
"""
# 下載 NASDAQ 指數資料（可選）
nasdaq = yf.Ticker("^IXIC")
hist = nasdaq.history(period="1y")
print(hist.head())
hist.to_csv("NASDAQ_data.csv")
"""

# 固定隨機種子，確保資料切割一致
SEED = 3
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

# 工具函式
def readData(f):
    return np.genfromtxt(f, delimiter=',', dtype=str)[1:]

def saveModel(net, optimizer, path):
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'close_mean': close_mean,
        'close_std': close_std
    }, path)

# 自定義 Dataset 類別
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='gpu'):
        self.data = data.to(device)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

def preprocess(data, flip=True):
    global close_mean, close_std

    numeric = np.array([row[1:] for row in data]).astype(np.float64)

    # 欄位索引說明
    # 0~4: NVDA (Close, High, Low, Open, Volume)
    # 5~9: IXIC
    # 10~14: NDX
    # 15~19: QQQ
    # 20~23: TW0050 & TW2330 (to remove)
    # 24~: sentiment

    # 移除台灣股價（20~23）
    numeric = np.delete(numeric, [20, 21, 22, 23], axis=1)

    if flip:
        numeric = np.flip(numeric, axis=0)

    close = numeric[:, 0]
    close_mean = close.mean()
    close_std = close.std()

    # 標準化所有欄位
    def standardize(x):
        return (x - x.mean()) / x.std()

    for i in range(numeric.shape[1]):
        if i == 0:
            numeric[:, i] = (numeric[:, i] - close_mean) / close_std
        else:
            numeric[:, i] = standardize(numeric[:, i])

    return numeric

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
    def __init__(self, feature_dim, output_len=1, num_layers=2, dropout=0, bidirectional=False):
        super(LSTMPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size=feature_dim, hidden_size=64, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = 64 * (2 if bidirectional else 1)
        
        self.model = nn.Sequential(
            nn.Linear(lstm_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            #nn.Dropout(0.5),
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.SiLU(),
            #nn.Dropout(0.4),
            nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.SiLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, output_len)
        )
        
        #nn.ReLU(),
        #nn.LeakyReLU(),
        #nn.SiLU(),
        
        #nn.Dropout(0.2),

    def predict(self, input):
        self.eval()  # 加這行避免 BatchNorm 報錯
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
            saveModel(net, optimizer, path)  # 儲存最佳模型
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
data = readData("./data/main_train_data.csv")
print("Num of samples:", len(data))
features = preprocess(data)
# 切分資料集：80% 訓練，20% 測試
train_full, test = train_test_split(features)

# 從訓練資料中切出 20% 做驗證集，但保留完整訓練資料不減少
valid_size = int(len(train_full) * 0.2)
valid = train_full[:valid_size]
train = train_full[valid_size:]
look_back = 5
forecast_horizon = 5
feature_dim = features.shape[1]
trainX, trainY = transform_dataset(train, look_back, forecast_horizon)
validX, validY = transform_dataset(valid, look_back, forecast_horizon)
testX, testY = transform_dataset(test, look_back, forecast_horizon)
trainset = Dataset(trainX, trainY, device)
validset = Dataset(validX, validY, device)
testset = Dataset(testX, testY, device)
batch_size = 256
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
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
    trainer(net, criterion, optimizer, trainloader, validloader, epoch_n=250, path=checkpoint)
else:
    checkpoint_data = torch.load(checkpoint)
    net.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    close_mean = checkpoint_data['close_mean']
    close_std = checkpoint_data['close_std']

print("close_mean / close_std:", close_mean, close_std)

# 測試模型
test_loss = tester(net, criterion, testloader)

# 預測結果先收集
net.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels, _ in testloader:
        outputs = net(inputs)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# 還原標準化結果
y_true = inverse_transform(np.array(y_true).flatten())
y_pred = inverse_transform(np.array(y_pred).flatten())

# 自動計算 RMSE（還原後單位）與 MAE
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
rmse = math.sqrt(test_loss) * close_std
print("Test MSELoss:", test_loss)
print("Test MAELoss:", mae * mae / rmse**2)
print("Test RMSE (actual price scale):", rmse)
print("Test RMAE (actual price scale):", mae)

net.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels, _ in testloader:
        outputs = net(inputs)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# 還原標準化結果
y_true = inverse_transform(np.array(y_true).flatten())
y_pred = inverse_transform(np.array(y_pred).flatten())

plt.figure(figsize=(10, 5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('True vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show(block=False)

# 單筆預測範例（兩個 feature：收盤價 + 成交量）
sample_data = readData("./data/main_test_data.csv")
sample_features = preprocess(sample_data, flip=False)
sample_input = torch.tensor(np.expand_dims(sample_features, axis=0), dtype=torch.float32).to(device)
predict = net.predict(sample_input)
print("Predicted Next 5 Days:", predict)

"""
Epoch 109 | Train Loss: 0.007649 | Val Loss: 0.106756
Epoch 110 | Train Loss: 0.007710 | Val Loss: 0.098730
Early stopping triggered.
Finished Training
close_mean / close_std: 77.37342913570508 42.56908416354035
Test MSELoss: 0.044703926891088486
Test MAELoss: 0.7469502998919839
Test RMSE (actual price scale): 9.000510521111476
Test RMAE (actual price scale): 7.778807009823642
"""