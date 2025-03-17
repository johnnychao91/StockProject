from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ✅ 設定 Selenium 選項
options = Options()
options.add_argument("--headless")  
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# ✅ 使用 `webdriver-manager` 下載 ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# ✅ 取得 Chrome 路徑與版本
chrome_version = driver.capabilities["browserVersion"]
chromedriver_version = driver.capabilities["chrome"]["chromedriverVersion"].split(" ")[0]
chrome_path = driver.capabilities.get("chrome", {}).get("binary", "未知")

print(f"✅ ChromeDriver 版本: {chromedriver_version}")
print(f"✅ Chrome 版本: {chrome_version}")
print(f"✅ Chrome 路徑: {chrome_path}")

driver.quit()
