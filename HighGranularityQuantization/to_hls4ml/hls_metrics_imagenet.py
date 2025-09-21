import hls4ml
from tensorflow import keras
from hls4ml.converters import convert_from_keras_model

# Load pretrained Keras model
keras_model = keras.applications.MobileNetV2(weights="imagenet", include_top=True)

# Convert to hls4ml model
hls_model = convert_from_keras_model(
    keras_model,
    output_dir="hls_prj_test",
    io_type="io_parallel"
)

# Print basic config
print("HLS Model Config:")
print(hls_model.config)

# Print layer summary
print("\nLayer Summary:")
for layer in hls_model.get_layers():
    print(f"{layer.name:30} type={layer.class_name:20} reuse={layer.attributes.get('ReuseFactor', 'N/A')}")

# Estimate latency (requires build/compile)
hls_model.build(reset=True)
hls_model.compile()

print("\nLatency Estimates:")
print(hls_model.config.get('Model', {}))

