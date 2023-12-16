from io import BytesIO
from PIL import Image
from urllib import request

import numpy as np

CLASSES = ['Covid', 'Normal', 'Viral Pneumonia']

def download_image(url):
  with request.urlopen(url) as resp:
    buffer = resp.read()
  stream = BytesIO(buffer)
  img = Image.open(stream)
  return img

def prepare_image(img, target_size):
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img = img.resize(target_size, Image.NEAREST)
  return img

def prepare_input(x):
  return x / 255.0

def get_prediction(url, interpreter, i_idx, o_idx):
  img = download_image(url)
  img = prepare_image(img, target_size=(299, 299))

  x = np.array(img, dtype='float32')
  X = np.array([x])
  X = prepare_input(X)

  interpreter.set_tensor(i_idx, X)
  interpreter.invoke()

  preds = interpreter.get_tensor(o_idx)
  preds = preds[0].tolist()
  preds_dict = dict(zip(CLASSES, preds))

  return preds_dict