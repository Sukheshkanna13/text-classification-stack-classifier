from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.base_classifiers = [
            ('nb', MultinomialNB()),
            ('rf', RandomForestClassifier())
        ]
        self.meta_classifier = MultinomialNB()
        self.stacked_classifier = StackingClassifier(
            estimators=self.base_classifiers,
            final_estimator=self.meta_classifier
        )

    def train(self, X_train, y_train):
        # Vectorize text data
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        
        # Train stacked classifier
        self.stacked_classifier.fit(X_train_transformed, y_train)

    def predict(self, X_test):
        # Vectorize test data
        X_test_transformed = self.vectorizer.transform(X_test)
        
        # Predict using stacked classifier
        return self.stacked_classifier.predict(X_test_transformed)


# Load your dataset
texts = [
    "Lionel Messi scored a hat-trick in the football match.",  # Replace "Text sample 1" with your actual text sample
    "Serena Williams won the tennis championship.",
    "LeBron James led his team to victory in the basketball game",
    # Add more text samples as needed
]

categories = [
    "football",  # Replace "Category 1" with the corresponding category for text sample 1
    "tennis",
    "basketball",
    # Add corresponding categories for text samples
]

# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, categories, test_size=0.2, random_state=42)

# Initialize your text classifier
classifier = TextClassifier()

# Train your classifier
classifier.train(X_train, y_train)

# Make predictions on testing data
y_pred = classifier.predict(X_test)

# Evaluate classifier performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

