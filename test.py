import pandas as pd
import feedparser
import os
from datetime import datetime, timedelta

# 搜索關鍵字
query = "nvidia"
# query = "tsmc"
# query = "nasdaq"
# query = "trump"

# 設定查詢時間範圍
start_date = datetime(2023, 1, 1)  # 開始日期
end_date = datetime.today()  # 結束日期

# 設定存儲目錄
base_dir = f"./data/google_news/{query}/"
os.makedirs(base_dir, exist_ok=True)  # 確保資料夾存在

# 逐日獲取新聞
current_date = start_date
while current_date <= end_date:
    # 設定 CSV 檔案名稱
    file_name = current_date.strftime("%Y_%m_%d") + ".csv"
    file_path = os.path.join(base_dir, file_name)

    # **檢查是否已存在檔案，若存在則跳過**
    if os.path.exists(file_path):
        print(f"{file_path} 已存在，跳過...")
        current_date += timedelta(days=1)
        continue  # 跳過當天，直接進入下一天

    # 設定 `after` 和 `before`（只查詢當日）
    after_str = current_date.strftime("%Y-%m-%d")
    before_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Google News RSS URL（查詢當日新聞）
    rss_url = f"https://news.google.com/rss/search?q={query}+after:{after_str}+before:{before_str}&hl=en-US&gl=US&ceid=US:en"

    # 解析 RSS
    news_feed = feedparser.parse(rss_url)

    # 解析新聞資料
    news_data = []
    articles = news_feed.entries[:100]  # 限制每天最多 100 篇
    for entry in articles:
        try:
            # 解析發佈時間
            published_time = entry.published if "published" in entry else "未知"

            # 轉換成 datetime 格式
            if published_time != "未知":
                pub_date = datetime.strptime(published_time, "%a, %d %b %Y %H:%M:%S %Z")
            else:
                pub_date = None

            # **印出完整的新聞資訊**
            print("\n🔍 發現新聞:")
            print(f"標題: {entry.get('title', 'N/A')}")
            print(f"發佈時間: {pub_date.strftime('%Y-%m-%d %H:%M:%S') if pub_date else '未知'}")
            print(f"描述: {entry.get('summary', 'N/A')}")
            print(f"來源: {entry.get('source', {}).get('title', '未知') if 'source' in entry else '未知'}")
            print(f"連結: {entry.get('link', 'N/A')}")
            print(f"GUID: {entry.get('id', 'N/A')}")
            print(f"類別: {entry.get('category', 'N/A')}")
            print(f"標籤: {[tag['term'] for tag in entry.get('tags', [])] if 'tags' in entry else 'N/A'}")
            print(f"圖片: {entry.get('media_content', 'N/A')}")
            print(f"更新時間: {entry.get('updated', 'N/A')}")
            print("-" * 80)

            # 檢查新聞時間是否在範圍內
            if pub_date and start_date <= pub_date <= end_date:
                news_data.append({
                    "發佈時間": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "標題": entry.get("title", "N/A"),
                    "描述": entry.get("summary", "N/A"),
                    "來源": entry.get("source", {}).get("title", "未知") if "source" in entry else "未知",
                    "連結": entry.get("link", "N/A"),
                    "GUID": entry.get("id", "N/A"),
                    "類別": entry.get("category", "N/A"),
                    "標籤": [tag["term"] for tag in entry.get("tags", [])] if "tags" in entry else "N/A",
                    "圖片": entry.get("media_content", "N/A"),
                    "更新時間": entry.get("updated", "N/A"),
                })

        except Exception as e:
            print(f"獲取 {after_str} 的新聞時出錯: {e}")

    # 如果當天有新聞，則存入 CSV
    if news_data:
        # 轉換為 DataFrame
        df = pd.DataFrame(news_data)

        # 儲存 CSV（避免編碼問題，使用 utf-8-sig）
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"✅ 新聞已儲存為 {file_path}")

    else:
        print(f"⚠️ {after_str} 沒有新聞可儲存")

    # 進入下一天
    current_date += timedelta(days=1)
