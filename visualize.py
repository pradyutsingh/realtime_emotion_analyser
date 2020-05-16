import tensorflow as tf

from cnn_model import define_model

model = define_model()

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_layer_names=True
)