import pandas as pd
import os
from datetime import datetime

# 讀取四個股價檔案
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

# 新增情感分析資料來源類別
categories = ['nvidia', 'gpu', 'nasdaq', 'trump', 'tsmc']
sentiment_base_path = './data/google_news_sentiment/'

# 將日期轉換成 datetime 以便處理
merged['Date'] = pd.to_datetime(merged['Date'])

# 建立情感欄位的預設值
for cat in categories:
    for col in ['Title_emotional_polarity', 'Title_subjectivity', 'content_emotional_polarity', 'content_subjectivity']:
        merged[f'{cat}_{col}'] = None

# 對每一列根據日期去讀取該天的各分類新聞情感檔案
for i, row in merged.iterrows():
    date_str = row['Date'].strftime('%Y_%m_%d')
    for cat in categories:
        file_path = os.path.join(sentiment_base_path, cat, f"{date_str}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                merged.at[i, f'{cat}_Title_emotional_polarity'] = df['標題_情感極性'].mean()
                merged.at[i, f'{cat}_Title_subjectivity'] = df['標題_主觀性'].mean()
                merged.at[i, f'{cat}_content_emotional_polarity'] = df['全文_情感極性'].mean()
                merged.at[i, f'{cat}_content_subjectivity'] = df['全文_主觀性'].mean()

# 輸出最終合併結果
merged.to_csv("./data/main_data.csv", index=False)

# 回傳合併後的資料
merged.head()

print("合併完成，已儲存為 main_data.csv")