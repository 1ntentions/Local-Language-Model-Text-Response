# Web Scraping
This portion of my project scrapes headlines from the business sections of two websites, Business Insider and CNBC, using Selenium.

## Steps to use
### Install Selenium
- This can be done by running the command "pip install selenium" in your computer's terminal  
- Selenium is used in this program for opening the Chrome browser (webdriver.Chrome()), navigating to each URL (driver.get()), finding specific HTML elements (driver.find_elements()), extracting data from those elements (.text, .get_attribute()), then closing the browser (driver.quit())
- Import the required modules from this package into a python program using "from selenium import webdriver", "from selenium.webdriver.chrome.service import Service", "from selenium.webdriver.common.by import By" (see my scraper.py file)
### Install the webdriver-manager package
- This can be done by running the command "pip install webdriver-manager" in your computer's terminal
- The webdriver-manager is used in this program to install and manage the correct version of the ChromeDriver so that Selenium can use it
- Import the required modules from this package into a python program using "from webdriver_manager.chrome import ChromeDriverManager" (see my scraper.py file)
