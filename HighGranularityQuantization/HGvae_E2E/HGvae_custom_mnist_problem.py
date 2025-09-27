import numpy as np
import keras
from HGvae_custom import VariationalAutoEncoder
import tensorflow as tf

# ===== Dataset (MNIST) =====
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# ===== VAE configuration =====
config = {
    "encoder_config": {"nodes": [256]},  # encoder hidden layer sizes
    "decoder_config": {"nodes": [256, 784]},  # decoder hidden layers + output
    "latent_dim": 16,
    "features": 784,
    "alpha": 1.0,  # reconstruction loss weight
    "beta": 1e-6,  # EBOP beta for HGQ
    "ap_fixed_kernel": 8,
    "ap_fixed_activation": 3,
    "ap_fixed_data": 8,
}

# ===== Define loss functions =====
reco_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="sum")
kl_loss = lambda z_mean, z_log_var: -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
)

# ===== Instantiate the VAE =====
vae = VariationalAutoEncoder(config, reco_loss, kl_loss)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# ===== Verification print =====
print("Class type:", type(vae))
print("Base classes:", vae.__class__.__bases__)
print("Is instance of keras.Model?", isinstance(vae, tf.keras.Model))
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
print("\nLayers inside VAE:")
for layer in vae.layers:
    print(f"  - {layer.name} ({layer.__class__.__name__})")

#vae.summary()


# ===== Optional callbacks =====
from HGQ import ResetMinMax, FreeBOPs
from keras.callbacks import LambdaCallback

pixel_loss_cb = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f"Epoch {epoch+1}: avg per-pixel loss = {logs['loss'] / config['features']:.6f}"
    )
)

callbacks = [ResetMinMax(), FreeBOPs(), pixel_loss_cb]

# ===== Train =====
vae.fit(
    x_train, x_train,
    validation_data=(x_test, x_test),
    epochs=10,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# ===== Cut model at latent mean =====
from tensorflow.keras import Model

encoder_to_mean = Model(
    inputs=vae.encoder.input,
    outputs=vae.encoder.get_layer("latent_mean").output
)

print("\n[INFO] Encoder-to-mean model summary:")
encoder_to_mean.summary()

