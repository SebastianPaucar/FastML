import keras
import numpy as np
from hgq.layers import QConv2D, QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.utils.sugar import FreeEBOPs

# Configure quantization
with QuantizerConfigScope(q_type='kif', place='weight', overflow_mode='SAT_SYM', round_mode='RND'):
    with QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='WRAP', round_mode='RND'):
        with LayerConfigScope(enable_ebops=False, beta0=0):
            model = keras.Sequential([
                keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),

                # Convolution + Pooling
                QConv2D(16, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2)),

                QConv2D(32, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2)),

                QConv2D(64, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Flatten(),
                QDense(128, activation='relu'),  # dense hidden layer
                QDense(10, activation='softmax')  # output probabilities
            ])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# EBOPs callback
ebops = FreeEBOPs()

# Train
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[ebops],
    verbose=2
)

