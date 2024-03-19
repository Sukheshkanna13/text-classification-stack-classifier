function classifyText() {
    const textInput = document.getElementById('textInput').value;
    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: textInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.category) {
            document.getElementById('result').innerText = `Predicted Category: ${data.category}`;
        } else {
            document.getElementById('result').innerText = 'Error: Unable to classify text.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error: Unable to process request.';
    });
}
