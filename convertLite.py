import tensorflow.lite as tlite

from tensorflow.keras.models import load_model

MODEL_PATH = 'models/xception_v4L_ep:79_val:0.951.h5'
LITE_NAME = 'x-rays-model.tflite'

model = load_model(MODEL_PATH)

converter = tlite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(LITE_NAME, 'wb') as f_out:
    f_out.write(tflite_model)