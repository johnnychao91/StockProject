import pandas as pd
import os
import time
import random
import requests
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from textblob import TextBlob
from tqdm import tqdm  # 進度條

# 設定 Selenium WebDriver
driver_path = "./mac/chromedriver"  # 修改為你的 chromedriver 路徑
options = Options()
#options.add_argument("--headless")  # 無頭模式
options.add_argument("--enable-gpu")
options.add_argument("--log-level=0")
options.add_argument("--window-size=1280,720")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-notifications")  # 直接關閉通知請求
options.add_argument("--disable-usb")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
#options.page_load_strategy = "eager"
options.page_load_strategy = "none"
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.execute_cdp_cmd("Page.setLifecycleEventsEnabled", {"enabled": True})

driver.set_page_load_timeout(10)
# 設定新聞類別
#category = "nvidia"
#category = "nasdaq"
category = "gpu"

# 設定新聞資料夾 & 儲存資料夾
news_dir = f"./data/google_news/{category}/"
output_dir = f"./data/google_news_sentiment/{category}/"
os.makedirs(output_dir, exist_ok=True)  # 確保輸出資料夾存在

# 設定開始 & 結束日期
start_date = datetime(2023, 1, 1)
end_date = datetime.today()

# 設定 requests session
session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 取得新聞原始連結（透過 Selenium）
def get_source_url(rss_url):
    try:
        #driver.get(rss_url)
        try:
            print(f"driver.get開始")
            driver.get(rss_url)
            print(f"driver.get完成")
        except TimeoutException:
            driver.execute_script("window.stop();")
            
        print(f"driver.get完成.")
        
        #wait = WebDriverWait(driver, timeout=30, poll_frequency=1)

        """
        # 檢查是否需要接受 Cookie
        if driver.current_url.startswith("https://consent.google.com/"):
            try:
                accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[.//span[text()="Accept all"]]')))
                accept_button.click()
                print("已接受 Cookies")
            except:
                print("無法找到接受 Cookies 按鈕")
        """
        
        # 等待跳轉至真實新聞頁面
        #wait.until(lambda d: not d.current_url.startswith("https://news.google.com/"))
        """
        try:
            wait.until(lambda d: not d.current_url.startswith("https://news.google.com/"))
        except TimeoutException:
            print("超時！未能從 Google News 跳轉到真實新聞網址")
            return None
        """
        #print("time.sleep(1)開始")
        #time.sleep(1)
        #print("time.sleep(1)完成")
        
        wait_for_page_load(driver)
        
        start_time = time.time()  # 記錄開始時間
        print(f"記錄開始時間: {start_time}")
        
        while time.time() - start_time < 120:
            print(f"get current_url")
            try:
                current_url = driver.current_url  # 取得當前網址
            except:
                continue
            print(f"current_url got")
            print(f"等待跳轉至真實新聞頁面: {current_url} ... {time.time() - start_time:.2f}")
            if not current_url.startswith("https://news.google.com/"):
                print(f"成功跳轉至: {current_url}")
                break  # 跳出迴圈
            time.sleep(0.5)  # ✅ 每 0.5 秒檢查一次
            #print(f"等待跳轉至真實新聞頁面: {current_url} ... {time.time() - start_time:.2f}")

        # 如果超時仍然沒跳轉
        if current_url.startswith("https://news.google.com/"):
            print("超時！未能從 Google News 跳轉到真實新聞網址")
            return None  # 設定為 None
            
        return driver.current_url

    except Exception as e:
        print(f"無法取得新聞原始網址: {e}")
        return None  # 若無法獲取網址，回傳 None

# 下載新聞全文（使用 requests + BeautifulSoup）
def fetch_article_content(url):
    if not url:
        return None  # 若沒有網址，回傳 None

    try:
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"{url} 無法獲取 (狀態碼: {response.status_code})")
            return None  # 若 HTTP 403/404 等，回傳 None

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 嘗試不同的標籤獲取新聞內容
        possible_tags = ["article", "div", "p"]
        for tag in possible_tags:
            content = soup.find_all(tag)
            if content:
                article_text = "\n".join([p.get_text() for p in content])
                return article_text.strip() if article_text.strip() else None

        return None  # 如果找不到新聞內容，回傳 None
    except Exception as e:
        print(f"解析新聞失敗: {e}")
        return None  # 若發生錯誤，回傳 None

# 定義情感分析函數（若遇到空值，返回 (0.0, 0.0)）
def analyze_sentiment(text):
    if not text or pd.isna(text) or text.strip() == "":
        return 0.0, 0.0  # 若無內容，返回中性情緒
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def wait_for_page_load(driver, timeout=120):
    """等待網頁完全載入 (`document.readyState == 'complete'`)，並顯示動畫"""
    start_time = time.time()
    dot_count = 0  # 用來控制 `.` 數量的變化
    while time.time() - start_time < timeout:
        page_state = driver.execute_script("return document.readyState;")

        # 產生動態的 `.` 數量（空格 → 1 → 2 → 3 → 空格 → 1 → 2 → 3）
        dot_count = (dot_count + 1) % 4  # 0, 1, 2, 3 循環
        dots = "." * dot_count + " " * (3 - dot_count)  # 空格填充，保持對齊

        print(f"\r等待頁面載入中{dots} ({page_state})", end="", flush=True)  # \r 讓字串覆蓋上一行
        #if page_state == "complete":
        if page_state == "interactive" or page_state == "complete":
            print("\n頁面載入完成")
            return True
        time.sleep(0.5)  # 每 0.5 秒檢查一次
    print("\n頁面載入超時")
    return False


# 逐日處理新聞
current_date = start_date
while current_date <= end_date:
    # 取得當天新聞 CSV 路徑
    date_str = current_date.strftime("%Y_%m_%d")
    input_file = os.path.join(news_dir, f"{date_str}.csv")
    output_file = os.path.join(output_dir, f"{date_str}.csv")

    # 如果輸出檔案已經存在，跳過
    if os.path.exists(output_file):
        print(f"{output_file} 已存在，跳過...")
        current_date += timedelta(days=1)
        continue

    # 檢查是否有該日期的新聞檔案
    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳過...")
        current_date += timedelta(days=1)
        continue

    print(f"處理 {input_file}...")

    # 讀取新聞 CSV
    df = pd.read_csv(input_file)

    # 確保必要欄位存在
    if "標題" not in df.columns or "連結" not in df.columns:
        print(f"{input_file} 缺少 '標題' 或 '連結' 欄位，跳過...")
        current_date += timedelta(days=1)
        continue

    # 用於存儲成功的新聞
    processed_data = []

    # 處理每一則新聞
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"下載 & 分析 {date_str}"):
        link = row["連結"]
        title = row["標題"]

        print(f"處理新聞: {link}")
        # 取得原始新聞連結
        source_url = get_source_url(link)
        if not source_url:
            print(f"無法獲取原始新聞連結，跳過該新聞: {title}")
            continue

        # 下載新聞全文
        full_text = fetch_article_content(source_url)
        if not full_text:
            print(f"無法獲取新聞全文，跳過該新聞: {source_url}")
            continue

        # 進行情感分析
        title_polarity, title_subjectivity = analyze_sentiment(title)
        text_polarity, text_subjectivity = analyze_sentiment(full_text)

        # 添加成功獲取的新聞數據
        processed_data.append([
            title, source_url, title_polarity, title_subjectivity, text_polarity, text_subjectivity
        ])

        # 防止 IP 被封鎖
        #time.sleep(3)
        wait_for_page_load(driver)

    # 檢查是否有可儲存的數據
    if processed_data:
        # 建立 DataFrame
        df_processed = pd.DataFrame(processed_data, columns=["標題", "真實新聞連結", "標題_情感極性", "標題_主觀性", "全文_情感極性", "全文_主觀性"])

        # 儲存結果
        df_processed.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"{output_file} 儲存完成！")
    else:
        print(f"{date_str} 沒有可儲存的新聞")

    # 進入下一天
    current_date += timedelta(days=1)

# 關閉 Selenium 瀏覽器
driver.quit()
print("所有新聞處理完畢，Selenium 已關閉！")
