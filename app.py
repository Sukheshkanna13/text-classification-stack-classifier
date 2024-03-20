from flask import Flask, request, jsonify
from classifier import TextClassifier

app = Flask(__name__)

classifier = TextClassifier()  # Create a classifier instance

# Assuming the downloaded dataset is named "spam.csv" and placed in the same directory
classifier.train("spam.csv")  # Train the model on the dataset

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json
        text = data['text']
        prediction = classifier.predict([text])[0]  # Predict for a single message
        label = "spam" if prediction == 1 else "not spam"  # Convert prediction to label
        return jsonify({'label': label}), 200
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Error: Unable to process request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
