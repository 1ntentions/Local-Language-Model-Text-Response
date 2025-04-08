from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service = Service(ChromeDriverManager().install(), options = options))

with open("urls.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]

all_headlines = []

for url in urls:
    driver.get(url)
    time.sleep(3)
    headlines = []

    if "businessinsider" in url:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.tout-image')
        headlines = [element.get_attribute("aria-label") for element in elements 
                     if element.get_attribute("aria-label")]
    elif "cnbc" in url:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.Card-title')
        headlines = [element.text.strip() for element in elements if element.text.strip()]

    all_headlines.extend(headlines)

with open("headlines.txt", "w", encoding = "utf-8") as out:
    for headline in all_headlines:
        out.write(f"{headline}\n")

driver.quit()