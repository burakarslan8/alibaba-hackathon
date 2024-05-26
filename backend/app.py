from flask import Flask, request, jsonify
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import io
import logging

app = Flask(__name__)

# configure logging
logging.basicConfig(level=logging.DEBUG)

# load inference model and feature extractor
model_name = "nateraw/food"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
inference_model = ViTForImageClassification.from_pretrained(model_name)

# load similarity model
similarity_model = SentenceTransformer('all-mpnet-base-v2')

# Handlers ---------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files or 'description' not in request.form:
            app.logger.error('No file or description part in the request')
            return jsonify({'error': 'No file or description part'}), 400

        image = request.files['file']
        description = request.form['description']

        if image:
            img_bytes = image.read()
            predictions = get_predictions(img_bytes)
            best_food = predictions[0]  # best predicted
            similarity_score = similarity(best_food[0], description)
            return jsonify({'predictions': [best_food], 'similarity_score': similarity_score})
    except Exception as e:
        app.logger.error(f'An error occurred: {e}')
        return jsonify({'error': str(e)}), 500
#--------------------------------------------------------------------------------
def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        app.logger.debug(f'Image mode: {image.mode}')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return feature_extractor(images=image, return_tensors="pt")
    except Exception as e:
        app.logger.error(f'Error processing image: {e}')
        raise

def get_predictions(image_bytes, top_k=5):
    try:
        inputs = transform_image(image_bytes)
        outputs = inference_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_labels = torch.topk(probs, top_k)
        top_probs = top_probs.squeeze().tolist()
        top_labels = top_labels.squeeze().tolist()
        return [(inference_model.config.id2label[label], prob) for label, prob in zip(top_labels, top_probs)]
    except Exception as e:
        app.logger.error(f'Error getting predictions: {e}')
        raise

def similarity(best_food_name, description):
    try:
        # 2 strings to compare
        string1 = best_food_name
        string2 = description

        if not string1 or not string2:
            return jsonify({'error': 'Both string1 and string2 are required'}), 400
        
        if "_" in string1:
            string1 = string1.replace("_", " ")

        # calculate the embeddings
        embedding1 = similarity_model.encode(string1, convert_to_tensor=True)
        embedding2 = similarity_model.encode(string2, convert_to_tensor=True)

        # calculate the cosine similarity
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

        return cosine_score.item()
    except Exception as e:
        app.logger.error('An error occurred in similarity calculation: %s', str(e))
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
