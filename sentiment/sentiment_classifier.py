from transformers import pipeline

class SentimentClassifier:
    def __init__(self, model_name, label_mapping = None):
        #Initializes sentiment analysis pipeline with specific model and tokenizer
        self.classifier = pipeline("sentiment-analysis", model = model_name, tokenizer = model_name)
        #Uses a custom label mapping or this provided one
        self.label_mapping = label_mapping or {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"}

    def classify_sentiment(self, txt):
        result = self.classifier(txt)[0]

        #Maps the label to an interpretable sentiment category
        return self.label_mapping.get(result['label'], "Neutral")