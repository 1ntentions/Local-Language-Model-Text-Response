# Financial News Sentiment Analyzer

A robust, object-oriented ETL (Extract, Transform, Load) pipeline that scrapes financial news headlines from major outlets and performs comparative sentiment analysis using state-of-the-art Hugging Face Transformers.

This project demonstrates scalable software engineering principles applied to data engineering tasks, utilizing the Strategy Pattern for scraping and separate micro-services for sentiment classification.

## Features

* **Modular Architecture:** Built using strict OOP principles. New news sources can be added by simply extending the `BaseScraper` abstract class without modifying the controller logic.
* **Headless Extraction:** Utilizes `Selenium` with Chrome in headless mode for efficient, background data mining.
* **ML-Powered Transformation:** Implements two distinct Transformer models to compare sentiment scoring:
    * **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`): Optimized for social/short-form text with 3-label classification (Positive, Neutral, Negative).
    * **GPT-2** (`mnoukhov/gpt2-imdb-sentiment-classifier`): A generative model adapted for binary sentiment classification.
* **Automated Testing:** Includes unit tests for both the scraping logic and the sentiment classification mapping.

## Project Structure

The project is organized into logical modules to ensure separation of concerns:

```text
financial-news-sentiment-analyzer/
├── scrapers/               # Extraction Layer
│   ├── base_scraper.py     # Abstract Base Class (ABC)
│   ├── cnbc_scraper.py     # CNBC Implementation
│   └── business_insider_scraper.py
├── sentiment/              # Transformation Layer (ML)
│   └── sentiment_classifier.py
├── main.py                 # Controller / Entry Point
├── urls.txt                # Configuration
├── requirements.txt        # Dependencies
└── tests/                  # Unit Tests

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iancdunn/Financial-News-Sentiment-Analyzer.git
    cd Financial-News-Sentiment-Analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure Sources:**
    Add target URLs to `urls.txt`. Currently supports *CNBC* and *Business Insider*.
    ```text
    [https://www.businessinsider.com/business](https://www.businessinsider.com/business)
    [https://www.cnbc.com/business/](https://www.cnbc.com/business/)
    ```

2.  **Run the Pipeline:**
    ```bash
    python main.py
    ```

3.  **View Outputs:**
    The pipeline generates two artifacts:
    * `headlines.txt`: Raw extracted text.
    * `sentiments.txt`: Sentiment analysis report comparing RoBERTa vs. GPT-2 classifications.

## Architecture & Design Patterns

### The Strategy Pattern (Scrapers)
The extraction layer uses the Strategy Pattern to handle different website structures. The `ScraperController` in `main.py` iterates through a list of URLs and dynamically instantiates the correct scraper class (`CNBCScraper` or `BusinessInsiderScraper`) at runtime. This allows the system to scale to hundreds of sources with minimal refactoring.

### Machine Learning Pipeline
The `SentimentClassifier` class acts as a wrapper around the Hugging Face `pipeline` API. It normalizes the output of different models (which often use different label keys like `LABEL_0` or `POS`) into a standardized human-readable format (`Positive`, `Negative`, `Neutral`).

## Testing

Run the included unit tests to verify the scrapers and model mappings:

```bash
pytest
