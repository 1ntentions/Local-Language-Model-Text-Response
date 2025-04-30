#Importing necessary modules for 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from scrapers.business_insider_scraper import BusinessInsiderScraper
from scrapers.cnbc_scraper import CNBCScraper
from sentiment.sentiment_classifier import SentimentClassifier

#Loads and executes individual scrapers for each website
class ScraperController:
    def __init__(self):
        self.scrapers = []
        self.all_headlines = []

    #Loads URLs from a file and initializes the appropriate scraper depending on the website
    def load_urls(self, fname):
        with open(fname, "r") as file:
            urls = [line.strip() for line in file if line.strip()]

        #Checks whether the URL is for Business Insider or CNBC
        for url in urls:
            if "businessinsider" in url:
                self.scrapers.append(BusinessInsiderScraper(url))
            elif "cnbc" in url:
                self.scrapers.append(CNBCScraper(url))

    #Executes the scrapers and collects the scraped headlines
    def scrape_all(self, driver):
        for scraper in self.scrapers:
            headlines = scraper.scrape(driver)
            self.all_headlines.extend(headlines)

    #Writes the collected headlines to a file, with one headline per line
    def write_headlines(self, fname):
        with open(fname, "w", encoding = "utf-8") as f:
            for headline in self.all_headlines:
                f.write(headline + "\n")

    #Returns all scraped headlines
    def get_headlines(self):
        return self.all_headlines

def main():
    # Sets up a headless Chrome browser for scraping without opening UI
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = options)

    #Scrapes news headlines
    controller = ScraperController()
    controller.load_urls("urls.txt")
    controller.scrape_all(driver)
    controller.write_headlines("headlines.txt")
    driver.quit()

    #RoBERTa's label mapping includes neutral, GPT 2's does not
    roberta = SentimentClassifier("cardiffnlp/twitter-roberta-base-sentiment", label_mapping = {
                                    "LABEL_0": "Negative",
                                    "LABEL_1": "Neutral",
                                    "LABEL_2": "Positive"})
    gpt2 = SentimentClassifier("mnoukhov/gpt2-imdb-sentiment-classifier", label_mapping = {
                                "LABEL_0": "Negative",
                                "LABEL_1": "Positive"})

    #Classifies the sentiment of each headline
    headlines = controller.get_headlines()
    roberta_sentiments = [roberta.classify_sentiment(headline) for headline in headlines]
    gpt2_sentiments = [gpt2.classify_sentiment(headline) for headline in headlines]

    #Writes sentiments to sentiments.txt file
    with open("sentiments.txt", "w") as file:
        file.write("---------SENTIMENTS FROM RoBERTa---------\n")
        for sentiment in roberta_sentiments:
            file.write(sentiment + "\n")
        file.write("---------SENTIMENTS FROM GPT 2---------\n")
        for sentiment in gpt2_sentiments:
            file.write(sentiment + "\n")
        
        print("Sentiments saved to sentiments.txt")

if __name__ == "__main__":
    main()