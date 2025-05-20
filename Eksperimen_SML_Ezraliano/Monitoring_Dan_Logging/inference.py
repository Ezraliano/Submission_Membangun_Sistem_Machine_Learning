from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.get_json()
    # Lakukan inference di sini
    result = {'prediction': 'Data Scientist'}
    duration = time.time() - start_time
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
