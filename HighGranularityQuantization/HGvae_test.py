import tensorflow as tf
import keras
from keras import layers
from HGQ.layers import HQuantize, HDense, HActivation
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf
from HGQ import ResetMinMax, FreeBOPs


tf.config.optimizer.set_jit(True)

# ==========
# 1. Configuración de cuantizadores
# ==========
paq_conf = get_default_paq_conf()
paq_conf["init_bw"] = 3   # activations start at 3 bits

kq_conf_8 = get_default_kq_conf()
kq_conf_8["init_bw"] = 8  # encoder first layer

kq_conf_4 = get_default_kq_conf()
kq_conf_4["init_bw"] = 4  # decoder

latent_dim = 16

# ==========
# 2. Dataset (MNIST)
# ==========
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

print("Train:", x_train.shape, " Test:", x_test.shape)

# ==========
# 3. VAE components
# ==========

# Encoder
inputs = keras.Input(shape=(784,))
x = HQuantize(beta=0, paq_conf=paq_conf)(inputs)
x = HDense(256, activation="relu", beta=0, kq_conf=kq_conf_8)(x)

z_mean = HDense(latent_dim, beta=0, kq_conf=kq_conf_8)(x)
z_log_var = HDense(latent_dim, beta=0, kq_conf=kq_conf_8)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
d = HQuantize(beta=0, paq_conf=paq_conf)(decoder_inputs)
d = HDense(256, activation="relu", beta=0, kq_conf=kq_conf_4)(d)
d = HDense(784, activation="linear", beta=0, kq_conf=kq_conf_4)(d)
#outputs = HDense(784, activation="sigmoid", beta=0, kq_conf=kq_conf_4)(d)
#outputs = layers.Activation("sigmoid")(d)
outputs = HActivation("sigmoid", beta=0, paq_conf=paq_conf)(d)
decoder = keras.Model(decoder_inputs, outputs, name="decoder")
decoded = decoder(z)

# ==========
# 4. VAE model
# ==========
vae = keras.Model(inputs, decoded, name="quantized_vae")

# VAE loss
recon_loss = tf.reduce_sum(
    keras.losses.binary_crossentropy(inputs, decoded), axis=-1
)
kl_loss = -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer="adam")

vae.summary()

# ==========
# 5. Train
# ==========
callbacks = [ResetMinMax(), FreeBOPs()]
vae.fit(
    x_train,
    x_train,
    validation_data=(x_test, x_test),
    epochs=10,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# ==========
# 6. Sample generation
# ==========
print("\nGenerating samples...")
z_samples = tf.random.normal(shape=(10, latent_dim))
reconstructions = decoder.predict(z_samples)

print("Reconstructed sample shape:", reconstructions.shape)
