import numpy as np
import os
from PIL import Image

# Load data
data = np.load("mnist_compressed.npz")
X_train, y_train = data["train_images"], data["train_labels"]
X_test, y_test = data["test_images"], data["test_labels"]

# Output directories
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/test", exist_ok=True)

# Save train images
for i, (img, label) in enumerate(zip(X_train, y_train)):
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(f"images/train/{label:02d}_{i}.png")

# Save test images
for i, (img, label) in enumerate(zip(X_test, y_test)):
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(f"images/test/{label:02d}_{i}.png")

print("✅ All images exported to 'images/train/' and 'images/test/' folders.")
