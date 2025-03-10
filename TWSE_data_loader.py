import pandas as pd
import os
from glob import glob

stock_no = '2330'  

def load_stock_data(stock_no, data_path='./data/tw_stock/'):
    """
    讀取指定股票代號的所有 CSV 檔案，列出可用月份，並顯示每個檔案的第一筆資料。
    
    :param stock_no: 股票代號，例如 '0050'
    :param data_path: 存放 CSV 檔案的資料夾路徑
    :return: DataFrame，包含所有讀取的股票數據
    """
    stock_path = os.path.join(data_path, stock_no)
    
    if not os.path.exists(stock_path):
        print(f"目標資料夾 {stock_path} 不存在！")
        return None

    # 找到所有符合 YYYY_MM.csv 格式的檔案
    file_list = sorted(glob(os.path.join(stock_path, "*.csv")))
    
    if not file_list:
        print(f"沒有找到 {stock_no} 的歷史數據！")
        return None

    print(f"找到 {len(file_list)} 個檔案：")
    
    # 讀取所有 CSV 檔案並合併
    df_list = []
    preview_list = []  # 存放每個檔案的第一筆數據

    for file in file_list:
        df = pd.read_csv(file, encoding='utf-8-sig')
        df['檔案名稱'] = os.path.basename(file)  # 加入來源檔案名稱
        df_list.append(df)

        # 如果檔案內有資料，列出第一筆
        if not df.empty:
            preview_list.append(df.iloc[0])

        print(f"  - {os.path.basename(file)}，共 {len(df)} 筆數據")

    # 顯示每個檔案的第一筆數據
    if preview_list:
        preview_df = pd.DataFrame(preview_list)
        print("\n每個檔案的第一筆數據：")
        print(preview_df[['檔案名稱'] + list(preview_df.columns[:-1])])  # 調整欄位順序，讓檔名在最前

    # 合併所有數據
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        print("⚠ 沒有可用的數據！")
        return None

# 測試讀取股票數據
df = load_stock_data(stock_no)

if df is not None:
    print("\n整體數據預覽：")
    print(df.head())
