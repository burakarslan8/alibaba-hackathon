from flask import Flask, request, render_template, url_for
import requests
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PRE_UPLOADED_FOLDER = 'static/pre_uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PRE_UPLOADED_FOLDER'] = PRE_UPLOADED_FOLDER

# In-memory storage for products
products = []

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pre_uploaded_images = os.listdir(app.config['PRE_UPLOADED_FOLDER'])
    
    if request.method == 'POST':
        title = request.form['title']
        file = request.files.get('file')
        pre_uploaded = request.form.get('pre_uploaded')
        
        if file and file.filename != '':
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
                product = {
                    'title': title,
                    'image_url': uploaded_image_url,
                    'best_food': data['predictions'][0],
                    'similarity_score': data['similarity_score']
                }
                products.append(product)
                return render_template('index.html', pre_uploaded_images=pre_uploaded_images, products=products)
            except requests.exceptions.RequestException as e:
                return f'An error occurred: {e}'
            except ValueError:
                return f'Error decoding JSON response: {response.content}'
        elif pre_uploaded:
            try:
                pre_uploaded_image_path = os.path.join(app.config['PRE_UPLOADED_FOLDER'], pre_uploaded)
                with open(pre_uploaded_image_path, 'rb') as img_file:
                    response = requests.post('http://127.0.0.1:5000/predict', files={'file': img_file}, data={'description': title})

                response.raise_for_status()  # Raise an error for bad status codes
                data = response.json()

                # Pass the URL of the pre-uploaded image
                uploaded_image_url = url_for('static', filename='pre_uploaded_images/' + pre_uploaded)
                product = {
                    'title': title,
                    'image_url': uploaded_image_url,
                    'best_food': data['predictions'][0],
                    'similarity_score': data['similarity_score']
                }
                products.append(product)
                return render_template('index.html', pre_uploaded_images=pre_uploaded_images, products=products)
            except requests.exceptions.RequestException as e:
                return f'An error occurred: {e}'
            except ValueError:
                return f'Error decoding JSON response: {response.content}'
    
    # List pre-uploaded images
    return render_template('index.html', pre_uploaded_images=pre_uploaded_images, products=products)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
