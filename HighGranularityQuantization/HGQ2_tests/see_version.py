import tensorflow as tf
import keras
import hgq

print("Keras version:",keras.__version__)  # Should be 3.x
print("TensorFlow version:", tf.__version__)
print(tf.sysconfig.get_build_info())
print(tf.config.list_physical_devices('GPU'))
print("hgq version:", hgq.__version__)

