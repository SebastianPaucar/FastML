# hgq_to_hls.py
import os
import tensorflow as tf
import keras
from HGQ.layers import HQuantize, HDense
from HGQ import ResetMinMax, FreeBOPs, trace_minmax, to_proxy_model
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf
import hls4ml

# ==========
# 1. Data (MNIST)
# ==========
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0
print(f"Train shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test shape: {x_test.shape}, Labels: {y_test.shape}\n")

# ==========
# 2. HGQ model
# ==========
paq_conf = get_default_paq_conf()
paq_conf['init_bw'] = 5   # start activations at 5 bits

kq_conf_8 = get_default_kq_conf()
kq_conf_8['init_bw'] = 10  # first dense layer with 10-bit weights

kq_conf_4 = get_default_kq_conf()
kq_conf_4['init_bw'] = 7   # second dense layer with 7-bit weights

print("Building HGQ model...")
model = keras.models.Sequential([
    HQuantize(beta=1e-5, input_shape=(784,), paq_conf=paq_conf),
    HDense(64, beta=1e-5, activation="relu", kq_conf=kq_conf_8),
    HDense(10, activation=None, beta=1e-5, kq_conf=kq_conf_4),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# ==========
# 3. Custom callback to log bitwidths
# ==========
class BitwidthLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} bitwidths:")
        for layer in self.model.layers:
            for var in layer.trainable_variables:
                if "bw" in var.name:
                    val = tf.reduce_mean(var).numpy()
                    print(f"  {var.name}: {val:.4f}")

# ==========
# 4. Train the model
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
# 5. Evaluate model
# ==========
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ==========
# 6. Convert to hls4ml
# ==========
print("\nCalibrating model for HLS conversion...")
calib_data = x_train[:2000]  # calibration subset
trace_minmax(model, calib_data, bsz=128, cover_factor=1.2)

print("Creating proxy model...")
proxy_model = to_proxy_model(model, aggressive=True)
proxy_model.summary()

print("Creating hls4ml configuration...")
hls_config = hls4ml.utils.config_from_keras_model(proxy_model, granularity='name')

# Optional: tweak ReuseFactor / Precision here if needed
# Example:
# hls_config['hdense']['ReuseFactor'] = 2
# hls_config['hdense']['Precision']['weight'] = 'ap_fixed<8,4>'

output_dir = 'hls_prj'
fpga_part = 'xc7z020clg484-1'  # replace with your target FPGA

print("\nConverting to hls4ml model...")
hls_model = hls4ml.converters.convert_from_keras_model(
    proxy_model,
    hls_config=hls_config,
    output_dir=output_dir,
    backend='vivado',  # or 'vitis'
    part=fpga_part
)

print("Building HLS project (C-simulation)...")
report = hls_model.build()
print("HLS project created in:", os.path.abspath(output_dir))

# ==========
# 7. Test predictions
# ==========
print("\nComparing predictions on first 5 test samples...")
preds = model.predict(x_test[:5])
print("Raw predictions (logits):")
print(preds)
print("Predicted classes:", tf.argmax(preds, axis=1).numpy())
print("True labels:      ", y_test[:5])

