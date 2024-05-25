from flask import Flask, request, jsonify, render_template
import requests
from PIL import Image
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_bytes = file.read()
            predictions = requests.post('http://127.0.0.1:5001/predict', files={'file': img_bytes})
            predictions = predictions.json()
            return render_template('index.html', predictions=predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)