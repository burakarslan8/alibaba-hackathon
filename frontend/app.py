from flask import Flask, request, render_template, url_for
import requests
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'title' not in request.form:
            return 'No file or title part'
        file = request.files['file']
        title = request.form['title']
        if file.filename == '':
            return 'No selected file'
        if file and title:
            try:
                # Ensure the upload folder exists
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                # Save the file to the static folder
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Now that the file is saved, open it for the POST request
                with open(file_path, 'rb') as img_file:
                    response = requests.post('http://127.0.0.1:5000/predict', files={'file': img_file}, data={'description': title})
                
                response.raise_for_status()  # Raise an error for bad status codes
                data = response.json()
                
                # Pass the URL of the uploaded image
                uploaded_image_url = url_for('static', filename='uploads/' + filename)
                return render_template('index.html', best_food=data['predictions'][0], similarity_score=data['similarity_score'], uploaded_image_url=uploaded_image_url, title=title)
            except requests.exceptions.RequestException as e:
                return f'An error occurred: {e}'
            except ValueError:
                return f'Error decoding JSON response: {response.content}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
