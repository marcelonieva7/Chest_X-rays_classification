import tflite_runtime.interpreter as tflite

MODEL_NAME = 'x-rays-model.tflite'

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

inp_index = interpreter.get_input_details()[0]['index']
out_index = interpreter.get_output_details()[0]['index']