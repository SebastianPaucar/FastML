import tensorflow as tf
import HGQ
import hls4ml

print("TensorFlow version:", tf.__version__)
print(tf.sysconfig.get_build_info())
print(tf.config.list_physical_devices('GPU'))
print("HGQ version:", HGQ.__version__)
print("hls4ml version:", hls4ml.__version__)
