import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
\
#stock_no = '0050'
stock_no = '2330'

# 設定開始與結束日期
today_str = datetime.today().strftime('%Y%m%d')

start_date = datetime.strptime('20210101', '%Y%m%d')
end_date = datetime.strptime(today_str, '%Y%m%d')

# 建立資料夾
save_path = './data/tw_stock/' + stock_no + '/'
os.makedirs(save_path, exist_ok=True)

# 逐月抓取資料
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y%m%d')
    file_name = current_date.strftime('%Y_%m') + '.csv'  # 變更檔案名稱格式為 YYYY_MM.csv
    
    # 發送請求獲取股票數據
    url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_no}'
    response = requests.get(url)
    content = json.loads(response.text)

    # 提取數據
    stock_data = content.get('data', [])
    col_name = content.get('fields', [])

    if stock_data:
        # 建立 DataFrame 並存成 CSV 檔案
        df = pd.DataFrame(data=stock_data, columns=col_name)
        file_path = os.path.join(save_path, file_name)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"CSV 檔案已儲存為 {file_path}")
    else:
        print(f"⚠ 無數據：{file_name}")

    # 移動到下一個月
    next_month = current_date.month + 1
    next_year = current_date.year + (next_month - 1) // 12
    next_month = (next_month - 1) % 12 + 1
    current_date = datetime(next_year, next_month, 1)
