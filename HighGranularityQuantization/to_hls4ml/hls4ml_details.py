import hls4ml
print("hls4ml version:", hls4ml.__version__)
from hls4ml.converters import convert_from_keras_model
print("convert_from_keras_model returns:", type(convert_from_keras_model))

