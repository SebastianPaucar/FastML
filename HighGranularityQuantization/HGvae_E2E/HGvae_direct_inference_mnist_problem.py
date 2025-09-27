import numpy as np
import tensorflow as tf
from keras.models import load_model

# ===== Import your custom layers =====
from HGQ import HDense, HQuantize   # make sure this path matches where HGQ.py is

# ===== Load the model =====
encoder_to_mean = load_model(
    "encoder_to_mean_model.h5",
    custom_objects={"HDense": HDense, "HQuantize": HQuantize}
)

print("[INFO] Model loaded successfully.")
encoder_to_mean.summary()

# ===== Example inference =====
# Generate dummy data shaped like MNIST (784 features per sample)
x_dummy = np.random.rand(1, 784).astype("float32")
z_mean = encoder_to_mean.predict(x_dummy)

print("[INFO] Latent mean vector shape:", z_mean.shape)
print(z_mean)

