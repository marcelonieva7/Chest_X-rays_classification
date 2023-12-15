from tensorflow.keras import Input, layers, Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def make_model(input_size=150,
              inner=False,
              inner_size=100,
              droprate=0,
              learning_rate=0.01
):
  base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(input_size, input_size, 3)
  )
  base_model.trainable = False

  inputs = Input(shape=(input_size, input_size, 3))
  base = base_model(inputs, training=False)

  vectors = layers.GlobalAveragePooling2D()(base)

  if inner:
    inner = layers.Dense(inner_size, activation='relu')(vectors)
    drop = layers.Dropout(droprate)(inner)
    outputs = layers.Dense(3)(drop)
  else:
    outputs = layers.Dense(3)(vectors)

  model = Model(inputs, outputs)
  optimizer = Adam(learning_rate=learning_rate)
  loss = CategoricalCrossentropy(from_logits=True)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
  )

  return model
