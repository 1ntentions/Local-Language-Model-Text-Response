from abc import ABC, abstractmethod

class BaseScraper(ABC):
    def __init__(self, url):
        self.url = url

    @abstractmethod
    def scrape(self, driver):
        pass