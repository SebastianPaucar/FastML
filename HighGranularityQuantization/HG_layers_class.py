from HGQ.layers import HQuantize, HDense, HActivation
from HGQ.utils.utils import get_default_kq_conf, get_default_paq_conf

# Example: create a quantized layer
paq_conf = get_default_paq_conf()
kq_conf_8 = get_default_kq_conf()

layer = HDense(256, activation="relu", beta=0, kq_conf=kq_conf_8)

# Print all attributes and methods
print("==== Attributes and Methods of the layer ====")
for attr in dir(layer):
    print(attr)

# Optional: inspect only "public" attributes (no underscores)
print("\n==== Public attributes ====")
for attr in dir(layer):
    if not attr.startswith("_"):
        print(attr)

# Optional: check if specific known attributes exist
for attr_name in ["bitwidths", "bops", "_bitwidth", "current_bops"]:
    if hasattr(layer, attr_name):
        print(f"\nFound attribute '{attr_name}':", getattr(layer, attr_name))

