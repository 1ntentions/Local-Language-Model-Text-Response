from sentiment.sentiment_classifier import SentimentClassifier

def test_sentiment_classifier_label_mapping():
    #Initializes the classifier with the GPT 2 model and its custom label mapping
    classifier = SentimentClassifier(
        model_name = "mnoukhov/gpt2-imdb-sentiment-classifier",
        label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Positive"
        }
    )
    
    #Classifies the sentiment of a basic positive sentence
    res = classifier.classify_sentiment("I love CS 325.")

    assert res in ["Positive", "Negative"], "Sentiment should be correctly mapped to Positive or Negative"