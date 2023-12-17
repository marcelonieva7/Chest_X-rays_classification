from flask import Flask
from flask import request, jsonify, render_template
from flasgger import Swagger

from server.model import interpreter, inp_index, out_index
from server.controllers import get_prediction

app = Flask('x-ray-classifier')
swagger = Swagger(app)

@app.route("/")
def home():
  return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
  """
  Endpoint for making predictions based on an input URL.

  ---
  tags:
    - Prediction
  parameters:
    - name: url
      in: body
      required: true
      schema:
          type: object
          properties:
            url:
              type: string
              description: URL of the image to be classified.
              example: "https://raw.githubusercontent.com/marcelonieva7/Chest_X-rays_classification/main/test/data/normal.jpeg"

  responses:
    200:
      description: Prediction results in JSON format.
      schema:
        type: object
        properties:
          Covid:
            type: number
            description: Probability of the image belonging to the Covid class.
          Normal:
            type: number
            description: Probability of the image belonging to the Normal class.
          Viral Pneumonia:
            type: number
            description: Probability of the image belonging to the Viral Pneumonia class.
      examples:
        {
          "Covid": 1.671286940574646,
          "Normal": 3.7960562705993652,
          "Viral Pneumonia": 1.1494133472442627
        }
  """
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
