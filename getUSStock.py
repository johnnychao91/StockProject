import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# 設定多個股票代碼與對應檔案名（因為 ^ 符號不能直接用在檔名中）
stocks = {
    "^NDX": "NDX",       # NASDAQ 100
    "^IXIC": "IXIC",     # NASDAQ Composite
    "QQQ": "QQQ",        # Invesco QQQ
    "NVDA": "NVDA"       # Nvidia
}

# 設定時間區間
start_date = "2023-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# 確保資料夾存在
output_folder = "./data/us_stock/"
os.makedirs(output_folder, exist_ok=True)

# 下載每一支股票資料
for symbol, name in stocks.items():
    print(f"下載中：{symbol}")
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

    if data.empty:
        print(f"{symbol} 無資料，略過。")
        continue

    # 整理欄位與順序
    data.reset_index(inplace=True)
    data = data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # 儲存為 CSV
    csv_filename = os.path.join(output_folder, f"{name}.csv")
    data.to_csv(csv_filename, index=False)
    
        # **讀取 CSV，刪除不必要的行**
    with open(csv_filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # **刪除包含 "Ticker" 和 "Date,,,," 的行**
    clean_lines = [line for line in lines if not line.startswith(",")]
    
    # **將乾淨的數據寫回 CSV**
    with open(csv_filename, "w", encoding="utf-8") as file:
        file.writelines(clean_lines)
        
    print(f"已儲存：{csv_filename}")

print(f"全部下載完成，截止日期：{end_date}")
