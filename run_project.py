import os
import importlib
import sys

# -------------------------------
# STEP 0: Install Compatible Dependencies (Python 3.11)
# -------------------------------
required_versions = {
    "tensorflow": "2.15.0",
    "numpy": "1.26.4",
    "pillow": "10.3.0",
    "matplotlib": "3.8.4",
    "seaborn": "0.13.2",
    "scikit-learn": "1.4.2"
}

print("\n📦 Checking and installing compatible libraries for Python 3.11...\n")
for pkg, version in required_versions.items():
    try:
        importlib.import_module(pkg if pkg != "pillow" else "PIL")
        print(f"✅ {pkg} is already installed.")
    except ImportError:
        print(f"⏳ Installing {pkg}=={version} ...")
        os.system(f"pip install {pkg}=={version}")

# Special check for tkinter (usually built-in)
try:
    import tkinter
    print("✅ tkinter is available.")
except ImportError:
    print("❌ tkinter is not available (should be built-in with Python).")

# -------------------------------
# STEP 1: Export Dataset from .npz
# -------------------------------
if not os.path.exists("images/train"):
    print("\n📁 Exporting dataset from mnist_compressed.npz...")
    os.system("python exporting_images_from_npz.py")
else:
    print("📁 Dataset already exported.")

# -------------------------------
# STEP 2: Sort Images into Folders
# -------------------------------
if not os.path.exists(os.path.join("images/train", "00")):
    print("📂 Sorting images into label folders...")
    os.system("python auto_sort.py")
else:
    print("📂 Images already sorted.")

# -------------------------------
# STEP 3: Train Model if Not Exists
# -------------------------------
if not os.path.exists("cnn_model_from_images.h5"):
    print("🧠 Training CNN model...")
    os.system("python train_cnn.py")
else:
    print("🧠 Model already trained. Skipping training.")

# -------------------------------
# STEP 4: Launch GUI
# -------------------------------
print("\n🚀 Launching Two-Digit Recognizer GUI...")
os.system("python two_digit_gui.py")
