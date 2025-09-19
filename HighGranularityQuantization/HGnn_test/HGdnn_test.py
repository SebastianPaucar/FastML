import tensorflow as tf
import keras
from HGQ.layers import HQuantize, HDenseBatchNorm, HDense
from HGQ import ResetMinMax, FreeBOPs
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Flatten
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf

# Activation quantizer config (pre-activation)
paq_conf = get_default_paq_conf()
paq_conf['init_bw'] = 5   # start activations at 3 bits

# Kernel quantizer config (weights)
kq_conf_8 = get_default_kq_conf()
kq_conf_8['init_bw'] = 10  # first dense layer with 8-bit weights

kq_conf_4 = get_default_kq_conf()
kq_conf_4['init_bw'] = 7  # second dense layer with 4-bit weights

# ==========
# 1. Datos (MNIST)
# ==========
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0
print(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels: {y_test.shape}\n")

# ==========
# 2. Modelo con HGQ
# ==========
print("Building HGQ model...")
model = keras.models.Sequential([
#    HQuantize(beta=1, name="quant_in", input_shape=(784,)),                # cuantización de entrada
    HQuantize(beta=1e-5, input_shape=(784,), paq_conf=paq_conf),
    HDense(64, beta=1e-5, activation="relu", kq_conf=kq_conf_8),    # capa densa cuantizada
    HDense(10, activation=None, beta=1e-5, kq_conf=kq_conf_4),    # otra capa cuantizada
#    Dense(10, activation="softmax")               # salida cuantizada
])

# Mostrar resumen del modelo
print("\nModel Summary:")
model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========
# 4. Inspect all trainable variables (once)
# ==========
print("\nListing all trainable variables (name, shape, min, max, mean):\n")
for var in model.trainable_variables:
    print(f"{var.name:40s} | shape: {var.shape} | min: {tf.reduce_min(var).numpy():.6f} | max: {tf.reduce_max(var).numpy():.6f} | mean: {tf.reduce_mean(var).numpy():.6f}")


# ==========
# 2b. Gradient check before training
# ==========
print("\nChecking gradient flow on a small batch...")

sample_x = x_train[:128]
sample_y = y_train[:128]

with tf.GradientTape() as tape:
    preds = model(sample_x, training=True)
    loss  = tf.keras.losses.sparse_categorical_crossentropy(sample_y, preds)

grads = tape.gradient(loss, model.trainable_weights)

for w, g in zip(model.trainable_weights, grads):
    if g is None:
        print(f"NO gradient for {w.name}")
    else:
        print(f"{w.name:40s} | grad norm = {tf.norm(g).numpy():.6f}")



# ==========
# 4. Custom callback to log bitwidths
# ==========
class BitwidthLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} bitwidths:")
        for layer in self.model.layers:
            for var in layer.trainable_variables:
                if "bw" in var.name:  # kernel_bw, activation_bw, etc.
                    val = tf.reduce_mean(var).numpy()
                    print(f"  {var.name}: {val:.4f}")



# ==========
# 3. Entrenamiento
# ==========
print("\nStarting training...")
callbacks = [ResetMinMax(), FreeBOPs(), BitwidthLogger()]
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# ==========
# 4. Evaluación
# ==========
print("\nEvaluating model on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ==========
# 5. Verificación extra
# ==========
print("\nChecking predictions on 5 test samples...")
sample_preds = model.predict(x_test[:5])
print("Raw predictions (logits or probabilities):")
print(sample_preds)
print("Predicted classes:", tf.argmax(sample_preds, axis=1).numpy())
print("True labels:      ", y_test[:5])

