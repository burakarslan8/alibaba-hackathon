from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# load the model
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/similarity', methods=['POST'])
def similarity():
    # 2 strings to compare
    data = request.get_json()
    string1 = data.get('string1')
    string2 = data.get('string2')

    if not string1 or not string2:
        return jsonify({'error': 'Both string1 and string2 are required'}), 400

    # calculate the embeddings
    embedding1 = model.encode(string1, convert_to_tensor=True)
    embedding2 = model.encode(string2, convert_to_tensor=True)

    # calculate the cosinus similarity
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    return jsonify({'similarity': cosine_score.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
