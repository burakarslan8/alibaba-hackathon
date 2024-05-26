from flask import Flask, request, render_template, url_for
import requests
import io
import os
from PIL import Image
import base64

app = Flask(__name__)

PRE_UPLOADED_FOLDER = 'static/pre_uploaded_images'
app.config['PRE_UPLOADED_FOLDER'] = PRE_UPLOADED_FOLDER

# in-memory storage for products
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
                # process the uploaded file in memory
                image = Image.open(file.stream)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                response = requests.post('http://127.0.0.1:5000/predict', files={'file': (file.filename, img_byte_arr, file.content_type)}, data={'description': title})
                response.raise_for_status()
                data = response.json()

                # pass the URL of the uploaded image (use a data URL for in-memory image)
                uploaded_image_url = f"data:{file.content_type};base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
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

                response.raise_for_status()
                data = response.json()

                # pass the URL of the pre-uploaded image
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
    
    # list pre-uploaded images
    return render_template('index.html', pre_uploaded_images=pre_uploaded_images, products=products)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=3000)
