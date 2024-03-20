import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = make_pipeline(self.vectorizer, LogisticRegression())

    def preprocess_text(self, text):
        # Lowercase and remove punctuation
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char == ' '])

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stop words (optional)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        return ' '.join(tokens)

    def load_data(self, data_path):
        """Loads data from CSV file into messages and labels"""
        df = pd.read_csv(data_path)
        return df['message'], df['label']

    def train(self, data_path):
        try:
            # Load data
            messages, labels = self.load_data(data_path)

            # Preprocess text data
            cleaned_messages = [self.preprocess_text(text) for text in messages]

            # Extract features
            features = self.vectorizer.fit_transform(cleaned_messages)

            # Train Logistic Regression classifier
            self.classifier.fit(features, labels)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise  # Re-raise the exception to be handled in app.py

    def predict(self, X_test):
        try:
            # Preprocess test data
            cleaned_test_messages = [self.preprocess_text(text) for text in X_test]

            # Extract features
            test_features = self.vectorizer.transform(cleaned_test_messages)

            # Make predictions
            predictions = self.classifier.predict(test_features)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
