import pandas as pd
import matplotlib.pyplot as plt

# 讀取使用者上傳的 CSV 檔案
file_path = "./data/NVDA_historical_data.csv"
df = pd.read_csv(file_path)

# 顯示前幾筆數據以檢查內容
df.head()

# 轉換日期格式
df['Date'] = pd.to_datetime(df['Date'])

# 設置圖表大小
plt.figure(figsize=(12, 6))

# 繪製收盤價趨勢圖
plt.plot(df['Date'], df['Close'], label='Close', color='blue')

# 設定標題與標籤
plt.title('NVIDIA (NVDA)')
plt.xlabel('Date')
plt.ylabel('Close (USD)')
plt.legend()
plt.grid()

# 顯示圖表
plt.show()
