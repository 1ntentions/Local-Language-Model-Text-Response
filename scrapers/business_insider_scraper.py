from scrapers.base_scraper import BaseScraper
from selenium.webdriver.common.by import By
import time

#Scraper class for Business Insider headlines that inherits from BaseScraper
class BusinessInsiderScraper(BaseScraper):
    def scrape(self, driver):
        driver.get(self.url)
        time.sleep(3)
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.tout-image')[:5]

        #Extracts and returns the text from the 'aria-label' attribute as headlines
        return [elem.get_attribute("aria-label") for elem in elements if elem.get_attribute("aria-label")]