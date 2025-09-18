import keras
import numpy as np
import tensorflow as tf
from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.utils.sugar import FreeEBOPs

# -----------------------------
# Quantization configuration
# -----------------------------
# Only quantize weights (activations remain float)
with QuantizerConfigScope(q_type='kif', place='weight', overflow_mode='SAT_SYM', round_mode='RND'):
    with LayerConfigScope(enable_ebops=False, beta0=0):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 image
            QDense(256, activation='relu'),
            QDense(128, activation='relu'),
            QDense(64, activation='relu'),
            QDense(10, activation='softmax')  # output probabilities
        ])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# -----------------------------
# Load MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# -----------------------------
# Gradient verification
# -----------------------------
# Take a small batch
x_small = x_train[:32]
y_small = y_train[:32]

with tf.GradientTape() as tape:
    logits = model(x_small, training=True)
    loss = keras.losses.sparse_categorical_crossentropy(y_small, logits)

# Compute gradients
grads = tape.gradient(loss, model.trainable_variables)

# Print summary
for i, g in enumerate(grads):
    print(f"Layer {i} gradient mean: {tf.reduce_mean(tf.abs(g)):.6f}")

# -----------------------------
# EBOPs callback (optional)
# -----------------------------
ebops = FreeEBOPs()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[ebops],
    verbose=2
)

