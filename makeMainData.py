import pandas as pd

# 讀取四個檔案
nvda = pd.read_csv("./data/us_stock/NVDA.csv")
ixic = pd.read_csv("./data/us_stock/IXIC.csv")
ndx = pd.read_csv("./data/us_stock/NDX.csv")
qqq = pd.read_csv("./data/us_stock/QQQ.csv")

# 重新命名欄位
nvda.columns = ['Date'] + [f"NVDA_{col}" for col in nvda.columns if col != 'Date']
ixic.columns = ['Date'] + [f"IXIC_{col}" for col in ixic.columns if col != 'Date']
ndx.columns = ['Date'] + [f"NDX_{col}" for col in ndx.columns if col != 'Date']
qqq.columns = ['Date'] + [f"QQQ_{col}" for col in qqq.columns if col != 'Date']

# 以 NVDA 的日期為基準進行合併
merged = nvda.merge(ixic, on="Date", how="left") \
             .merge(ndx, on="Date", how="left") \
             .merge(qqq, on="Date", how="left")

merged.to_csv("./data/main_data.csv", index=False)
print("合併完成，已儲存為 merged_stocks.csv")

# 回傳合併後的資料
merged.head()
