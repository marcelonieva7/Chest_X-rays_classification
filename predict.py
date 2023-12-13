from flask import Flask
from flask import request, jsonify

from server.model import interpreter, inp_index, out_index
from server.controllers import get_prediction

app = Flask('x-ray-classifier')

@app.route('/predict', methods=['POST'])
def predict():
  prediction = get_prediction(request.json['url'], interpreter, inp_index, out_index)
  return jsonify(prediction)

@app.route('/health', methods=['GET'])
def health_check():
  """
  A function that handles the '/health' route with a GET method.
  ---
  tags:
    - Health Check
  responses:
    200:
      description: API health status
      schema:
        type: string
        example: "API is healthy"
  """
  return 'API is healthy'

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=9696)
