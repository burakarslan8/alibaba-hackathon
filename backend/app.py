from flask import Flask, request, jsonify, render_template
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load model and feature extractor
model_name = "nateraw/food"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return feature_extractor(images=image, return_tensors="pt")

def get_predictions(image_bytes, top_k=5):
    inputs = transform_image(image_bytes)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_labels = torch.topk(probs, top_k)
    top_probs = top_probs.squeeze().tolist()
    top_labels = top_labels.squeeze().tolist()
    return [(model.config.id2label[label], prob) for label, prob in zip(top_labels, top_probs)]

@app.route('/predict', methods=['POST'])
def upload_file():
    
    file = request.files['file']
    if file:
        img_bytes = file.read()
        predictions = get_predictions(img_bytes)
        return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
