import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

from HGQ.layers import HQuantize, HDense, HActivation
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf

tf.config.optimizer.set_jit(True)

# ==========
# 1. Configuración de cuantizadores
# ==========
paq_conf = get_default_paq_conf()
paq_conf["init_bw"] = 3   # activations start at 3 bits

kq_conf = get_default_kq_conf()
kq_conf["init_bw"] = 8  # weights

# ==========
# 2. Dataset (MNIST)
# ==========
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

# ==========
# 3. Simple DNN model
# ==========
inputs = layers.Input(shape=(784,))
x = HQuantize(beta=0, paq_conf=paq_conf)(inputs)
x = HDense(256, activation="relu", beta=0, kq_conf=kq_conf)(x)
x = HDense(128, activation="relu", beta=0, kq_conf=kq_conf)(x)
x = HDense(10, activation="linear", beta=0, kq_conf=kq_conf)(x)
outputs = HActivation("softmax", beta=0, paq_conf=paq_conf)(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ==========
# 4. Callback to record BOPs & bitwidths
# ==========
class BopsBitwidthLogger(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.bops_history = []
        self.bitwidths_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # --- Trace min/max manually ---
        for x_batch, _ in self.dataset.take(5):  # a few batches
            _ = self.model(x_batch, training=False)
            for layer in self.model.layers:
                if hasattr(layer, "record_minmax"):
                    layer.record_minmax()

        # --- Compute BOPs and bitwidths ---
        total_bops = 0
        bitwidths = []
        for layer in self.model.layers:
            if hasattr(layer, "compute_exact_bops"):
                try:
                    total_bops += layer.compute_exact_bops()
                except AssertionError:
                    pass
            if hasattr(layer, "act_bw_exact"):
                try:
                    bw = layer.act_bw_exact
                    if isinstance(bw, tf.Tensor):
                        bw = bw.numpy()
                    bitwidths.append(np.mean(bw))
                except AssertionError:
                    pass

        self.bops_history.append(total_bops)
        self.bitwidths_history.append(np.mean(bitwidths) if bitwidths else 0)
        print(f"Epoch {epoch+1}: BOPs={total_bops:.0f}, Avg bitwidth={np.mean(bitwidths):.2f}")

# ==========
# 5. Train
# ==========
callbacks = [BopsBitwidthLogger(train_dataset)]
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=2)

# ==========
# 6. Plot performance
# ==========
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# BOPs and Bitwidth
plt.subplot(1,2,2)
plt.plot(callbacks[0].bops_history, label="BOPs")
plt.plot(callbacks[0].bitwidths_history, label="Avg Bitwidth")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Bitwidths & BOPs evolution")
plt.legend()

plt.tight_layout()
plt.savefig("dnn_bops_bitwidths.png", dpi=300)
plt.show()

