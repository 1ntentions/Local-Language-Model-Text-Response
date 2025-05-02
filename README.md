# News Headline Sentiment Classifier
This portion of my project scrapes headlines from the business sections of two websites, Business Insider and CNBC, using Selenium. Then, two models from HuggingFace, RoBERTa and GPT 2, are loaded to classify the sentiments of these scraped headlines and write them to the "sentiments.txt" file.  
You can find my docker repository for this on Docker Hub as 1ntentions/headline-sentiment
## Steps to use
### Install Torch
- This can be done by running the command ``` pip install torch ``` in your computer's terminal
- Torch is used in this program for
  1. Turning text inputs into PyTorch tensors
### Install Transformers
- This can be done by running the command ``` pip install transformers ``` in your computer's terminal
- Transformers is used in this program for
  1. Loading pretrained models, their tokenizers, processing text, inferring sentiments from it (```pipeline()```)
- Import examples in Python
  1. ``` from transformers import pipline ```
### Install Selenium
- This can be done by running the command ``` pip install selenium ``` in your computer's terminal  
- Selenium is used in this program for
    1. Opening the Chrome browser (```webdriver.Chrome()```)
    2. Navigating to each URL (```driver.get()```)
    3. Finding specific HTML elements (```driver.find_elements()```)
    4. Extracting data from those elements (```.text```, ```.get_attribute()```)
    5. Closing the browser (```driver.quit()```)
- Import examples in Python:
    1. ``` from selenium import webdriver ```
    2. ``` from selenium.webdriver.chrome.service import Service ```
    3. ``` from selenium.webdriver.common.by import By ```
### Install the webdriver-manager package
- This can be done by running the command ``` pip install webdriver-manager ``` in your computer's terminal
- The webdriver-manager is used in this program to
    1. Install and manage the correct version of the ChromeDriver so that Selenium can use it
- Import examples in Python:
    1. ``` from webdriver_manager.chrome import ChromeDriverManager ```
