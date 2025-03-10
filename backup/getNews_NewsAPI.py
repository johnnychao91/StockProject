import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from newspaper import Article
from tqdm import tqdm
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 可切換的新聞類別
category = "nasdaq"  # 可改為 "nvidia", "trump", "tsmc"

# 設定新聞資料夾 & 儲存資料夾
news_dir = f"./data/google_news/{category}/"
output_dir = f"./data/google_news_sentiment/{category}/"
os.makedirs(output_dir, exist_ok=True)

# 設定查詢範圍（開始日期 & 今天）
start_date = datetime(2023, 1, 1)
end_date = datetime.today()

# 初始化 VADER
vader = SentimentIntensityAnalyzer()

# 定義情感分析函數
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0, 0.0  # 若無內容，返回中性情緒
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# 使用 VADER 進行情感分析
def analyze_vader(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0  # 若無內容，返回中性情緒
    return vader.polarity_scores(text)["compound"]

# 下載新聞全文（如果 newspaper3k 失敗，改用 BeautifulSoup）
def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        # 嘗試用 requests + BeautifulSoup 下載
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                return "\n".join([p.get_text() for p in paragraphs])
        except:
            return None  # 若無法下載，回傳 None

# 逐日處理新聞
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y_%m_%d")
    input_file = os.path.join(news_dir, f"{date_str}.csv")
    output_file = os.path.join(output_dir, f"{date_str}.csv")

    if os.path.exists(output_file):
        print(f"{output_file} 已存在，跳過...")
        current_date += timedelta(days=1)
        continue

    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳過...")
        current_date += timedelta(days=1)
        continue

    print(f"處理 {input_file}...")

    df = pd.read_csv(input_file)

    if "標題" not in df.columns or "連結" not in df.columns:
        print(f"{input_file} 缺少 '標題' 或 '連結' 欄位，跳過...")
        current_date += timedelta(days=1)
        continue

    # 分析標題情感
    df["標題_情感極性"], df["標題_主觀性"] = zip(*df["標題"].apply(analyze_sentiment))
    df["標題_VADER_分數"] = df["標題"].apply(analyze_vader)  # 加入 VADER 分析

    # 下載 & 分析新聞全文
    full_texts = []
    full_sentiments = []
    for link in tqdm(df["連結"], desc=f"下載 & 分析 {date_str}"):
        full_text = fetch_article_content(link)
        full_texts.append(full_text)

        if full_text:
            polarity, subjectivity = analyze_sentiment(full_text)
        else:
            polarity, subjectivity = 0.0, 0.0  # 無法下載，預設 0

        full_sentiments.append((polarity, subjectivity))
        time.sleep(2)  # 避免 IP 被封鎖

    df["新聞全文"] = full_texts
    df[["全文_情感極性", "全文_主觀性"]] = pd.DataFrame(full_sentiments)

    # 選擇要輸出的欄位
    df = df[["標題", "標題_情感極性", "標題_主觀性", "標題_VADER_分數", "全文_情感極性", "全文_主觀性"]]

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"{output_file} 儲存完成！")

    current_date += timedelta(days=1)
