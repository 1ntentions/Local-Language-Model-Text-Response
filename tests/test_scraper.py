from scrapers.business_insider_scraper import BusinessInsiderScraper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def test_business_insider_scraper():
    url = "https://www.businessinsider.com/business"
    scraper = BusinessInsiderScraper(url)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = options)

    #Uses the scraper to extract headlines from the page
    headlines = scraper.scrape(driver)
    driver.quit()

    assert isinstance(headlines, list), "Scraper should return a list"
    assert all(isinstance(headline, str) for headline in headlines), "All headlines should be strings"
    assert len(headlines) > 0, "At least one headline should be returned"