# Required modules
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# Sets chrome options to run without a browser window, initializes chrome driver with these options
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service = Service(ChromeDriverManager().install(), options = options))

# Reads the two URLs from the urls.txt file
with open("urls.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip()]

all_headlines = []

# Loops through the two URLs from urls.txt and scrapes headlines based on website structure
for url in urls:
    driver.get(url)
    time.sleep(3)
    headlines = []

    # Scraping Business Insider
    if "businessinsider" in url:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.tout-image')
        headlines = [element.get_attribute("aria-label") for element in elements 
                     if element.get_attribute("aria-label")]
    # Scraping CNBC
    elif "cnbc" in url:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.Card-title')
        headlines = [element.text.strip() for element in elements if element.text.strip()]

    # Adds the scraped headlines to the overall list
    all_headlines.extend(headlines)

# Writes all headlines to headlines.txt, with one headline per line
with open("headlines.txt", "w", encoding = "utf-8") as out:
    for headline in all_headlines:
        out.write(f"{headline}\n")

driver.quit()
