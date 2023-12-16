from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils.train import make_model

train_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=10,
  zoom_range=0.15,
  horizontal_flip=True,
  brightness_range=[0.9, 1.1],
  shear_range=0.2,
  validation_split=0.25
)

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  validation_split = 0.25
)

train_ds = train_gen.flow_from_directory(
  './Covid19-dataset/train',
  target_size=(299, 299),
  batch_size=32,
  shuffle=False,
  seed=123,
  subset="training"
)

val_ds = val_gen.flow_from_directory(
  './Covid19-dataset/train',
  target_size=(299, 299),
  batch_size=32,
  shuffle=False,
  seed=123,
  subset="validation"
)

INP_SIZE = 299
LR = 0.001
INNER_SIZE = 10
DROP = 0.5

model = make_model(input_size=INP_SIZE,
                   learning_rate=LR,
                   inner=True,
                   inner_size=INNER_SIZE,
                   droprate=DROP
                  )

chechpoint = ModelCheckpoint(
  'models/xception_EP:{epoch:02d}_VAL:{val_accuracy:.3f}.h5',
  save_best_only=True,
  monitor='val_accuracy',
  mode='max'
)

es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01, verbose=1, patience=60)

model.fit(
  train_ds,
  epochs=150,
  validation_data=val_ds,
  callbacks=[chechpoint, es]
)