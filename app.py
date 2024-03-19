from flask import Flask, request, jsonify
from classifier import TextClassifier

app = Flask(__name__)
classifier = TextClassifier()

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json
        text = data['text']
        category = classifier.predict([text])
        return jsonify({'category': category}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)  # Run Flask app on port 5000
