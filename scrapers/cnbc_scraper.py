from scrapers.base_scraper import BaseScraper
from selenium.webdriver.common.by import By
import time

#Scraper class for CNBC headlines that inherits from BaseScraper
class CNBCScraper(BaseScraper):
    def scrape(self, driver):
        driver.get(self.url)
        time.sleep(3)
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.Card-title')[:5]

        #Returns the stripped text from the 'a.Card-title' CSS elements as headlines
        return [elem.text.strip() for elem in elements if elem.text.strip()]