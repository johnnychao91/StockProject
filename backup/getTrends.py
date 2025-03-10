import pandas as pd
from pytrends.request import TrendReq

# 初始化 pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# 設定要搜尋的關鍵字
kw_list = ["NVIDIA"]  # 你可以加入多個關鍵字，例如 ["NVIDIA", "AMD"]

# 取得趨勢數據（時間範圍：過去 1 年）
pytrends.build_payload(kw_list, timeframe="today 12-m", geo="US")  # 可調整地區

# 下載 Google 搜尋趨勢數據
df = pytrends.interest_over_time()

# 移除 "isPartial" 欄位（如果存在）
if "isPartial" in df.columns:
    df = df.drop(columns=["isPartial"])

# 儲存為 CSV
csv_filename = "./google_trends_nvidia.csv"
df.to_csv(csv_filename, encoding="utf-8-sig")
print(f"✅ Google 趨勢數據已儲存為 {csv_filename}")

# 顯示前 5 筆數據
print(df.head())
