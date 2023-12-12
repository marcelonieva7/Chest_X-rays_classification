from flask import Flask
from flask import request, jsonify, send_file

from server.controllers import get_prediction

import tflite_runtime.interpreter as tflite

MODEL_NAME = 'x-rays-model.tflite'

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

inp_index = interpreter.get_input_details()[0]['index']
out_index = interpreter.get_output_details()[0]['index']

app = Flask('x-ray-classifier')

@app.route('/predict', methods=['POST'])
def predict():
  prediction = get_prediction(request.json['url'], interpreter, inp_index, out_index)
  return jsonify(prediction)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=9696)