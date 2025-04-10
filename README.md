# Web Scraping
This portion of my project scrapes headlines from the business sections of two websites, Business Insider and CNBC, using Selenium.

## Steps to use
### Install Selenium
- This can be done by running the command "pip install selenium" in your computer's terminal  
- Selenium is used in this program for
    1. Opening the Chrome browser(webdriver.Chrome())
    2. Navigating to each URL (driver.get())
    3. Finding specific HTML elements (driver.find_elements())
    4. Extracting data from those elements (.text, .get_attribute())
    5. Closing the browser (driver.quit())
- Import examples in Python: "from selenium import webdriver", "from selenium.webdriver.chrome.service import Service", "from selenium.webdriver.common.by import By" (see my scraper.py file)
### Install the webdriver-manager package
- This can be done by running the command "pip install webdriver-manager" in your computer's terminal
- The webdriver-manager is used in this program to
    1. Install and manage the correct version of the ChromeDriver so that Selenium can use it
- Import examples in Python: "from webdriver_manager.chrome import ChromeDriverManager" (see my scraper.py file)
