from flask import Flask, request, render_template
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'description' not in request.form:
            return 'No file or description part'
        file = request.files['file']
        description = request.form['description']
        if file.filename == '':
            return 'No selected file'
        if file and description:
            try:
                response = requests.post('http://127.0.0.1:5000/predict', files={'file': file}, data={'description': description})
                response.raise_for_status()  # Raise an error for bad status codes
                predictions = response.json()
                return render_template('index.html', predictions=predictions)
            except requests.exceptions.RequestException as e:
                return f'An error occurred: {e}'
            except ValueError:
                return f'Error decoding JSON response: {response.content}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
