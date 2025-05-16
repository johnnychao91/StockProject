import pandas as pd
import os
from datetime import datetime

# 讀取四個股價檔案
nvda = pd.read_csv("./data/us_stock/NVDA.csv")
ixic = pd.read_csv("./data/us_stock/IXIC.csv")
ndx = pd.read_csv("./data/us_stock/NDX.csv")
qqq = pd.read_csv("./data/us_stock/QQQ.csv")

# 確保美股日期格式正確
nvda['Date'] = pd.to_datetime(nvda['Date'])
ixic['Date'] = pd.to_datetime(ixic['Date'])
ndx['Date'] = pd.to_datetime(ndx['Date'])
qqq['Date'] = pd.to_datetime(qqq['Date'])

# 讀取台股資料（0050 和 2330）
tw_0050_all = []
tw_2330_all = []
cols = ['日期', '收盤價', '成交筆數']

# 讀取 2023/01 到 2025/03 所有月份的台股資料
for year in range(2023, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 3:
            break
        month_str = f"{year}_{month:02d}"
        file_0050 = f"./data/tw_stock/0050/{month_str}.csv"
        file_2330 = f"./data/tw_stock/2330/{month_str}.csv"
        if os.path.exists(file_0050):
            df_0050 = pd.read_csv(file_0050)[cols]
            tw_0050_all.append(df_0050)
        if os.path.exists(file_2330):
            df_2330 = pd.read_csv(file_2330)[cols]
            tw_2330_all.append(df_2330)

# 合併與重新命名欄位
tw_0050 = pd.concat(tw_0050_all, ignore_index=True)
tw_0050.columns = ['Date', 'TW0050_Close', 'TW0050_Volume']
tw_2330 = pd.concat(tw_2330_all, ignore_index=True)
tw_2330.columns = ['Date', 'TW2330_Close', 'TW2330_Volume']

# 將收盤價與成交筆數轉為數值（移除逗號、雙引號與空白）
tw_0050['TW0050_Close'] = tw_0050['TW0050_Close'].astype(str).str.replace(',', '').str.replace('"', '').str.strip().astype(float)
tw_0050['TW0050_Volume'] = tw_0050['TW0050_Volume'].astype(str).str.replace(',', '').str.replace('"', '').str.strip().astype(float)
tw_2330['TW2330_Close'] = tw_2330['TW2330_Close'].astype(str).str.replace(',', '').str.replace('"', '').str.strip().astype(float)
tw_2330['TW2330_Volume'] = tw_2330['TW2330_Volume'].astype(str).str.replace(',', '').str.replace('"', '').str.strip().astype(float)

# 轉換日期格式（民國轉西元）
def convert_roc_date(roc_date):
    year, month, day = map(int, roc_date.split('/'))
    return f"{1911 + year:04d}-{month:02d}-{day:02d}"

tw_0050['Date'] = pd.to_datetime(tw_0050['Date'].apply(convert_roc_date))
tw_2330['Date'] = pd.to_datetime(tw_2330['Date'].apply(convert_roc_date))

# 填補缺失值：先依日期排序再用前值填補
all_dates = pd.DataFrame({'Date': nvda['Date']})
tw_0050 = pd.merge(all_dates, tw_0050, on='Date', how='left').sort_values('Date').ffill()
tw_2330 = pd.merge(all_dates, tw_2330, on='Date', how='left').sort_values('Date').ffill()

# 重新命名美股欄位
nvda.columns = ['Date'] + [f"NVDA_{col}" for col in nvda.columns if col != 'Date']
ixic.columns = ['Date'] + [f"IXIC_{col}" for col in ixic.columns if col != 'Date']
ndx.columns = ['Date'] + [f"NDX_{col}" for col in ndx.columns if col != 'Date']
qqq.columns = ['Date'] + [f"QQQ_{col}" for col in qqq.columns if col != 'Date']

# 合併所有資料
merged = nvda.merge(ixic, on="Date", how="left") \
             .merge(ndx, on="Date", how="left") \
             .merge(qqq, on="Date", how="left") \
             .merge(tw_0050, on="Date", how="left") \
             .merge(tw_2330, on="Date", how="left")

# 新增情感分析資料來源類別
categories = ['nvidia', 'gpu', 'nasdaq', 'trump', 'tsmc']
sentiment_base_path = './data/google_news_sentiment/'

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

# 拆分資料為 train/test
merged = merged.sort_values(by='Date')
train_data = merged.iloc[:-5]
test_data = merged.iloc[-5:]

# 輸出最終合併結果
train_data.to_csv("./data/main_train_data.csv", index=False)
test_data.to_csv("./data/main_test_data.csv", index=False)
print("合併完成，已儲存為 main_train_data.csv 與 main_test_data.csv")

# 回傳合併後的資料
train_data.head()
