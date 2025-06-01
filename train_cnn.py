import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Parameters
img_size = (128, 128)
batch_size = 64
num_classes = 100

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "images/train", target_size=img_size, color_mode="grayscale",
    class_mode="categorical", batch_size=batch_size, shuffle=True
)
test_data = test_gen.flow_from_directory(
    "images/test", target_size=img_size, color_mode="grayscale",
    class_mode="categorical", batch_size=batch_size, shuffle=False
)

# Total image count
train_total = sum([len(files) for r, d, files in os.walk("images/train")])
test_total = sum([len(files) for r, d, files in os.walk("images/test")])

# Bar chart - Dataset split
plt.figure(figsize=(6, 4))
plt.bar(["Train", "Test"], [train_total, test_total], color=["skyblue", "salmon"])
plt.title("Total Images: Train vs. Test")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("dataset_split_summary.png")

# Bar chart - Class distribution
train_class_counts = [len(os.listdir(os.path.join("images/train", cls))) for cls in sorted(os.listdir("images/train"))]
plt.figure(figsize=(12, 5))
plt.bar(range(100), train_class_counts)
plt.title("Training Set Class Distribution (00–99)")
plt.xlabel("Class Label")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig("class_distribution_train.png")

# Heatmap - Class imbalance
plt.figure(figsize=(20, 2))
sns.heatmap(np.array(train_class_counts).reshape(1, -1), cmap="YlGnBu", cbar=True, xticklabels=range(100))
plt.title("Heatmap of Class Distribution (Train Set)")
plt.yticks([])
plt.xlabel("Class Label")
plt.savefig("class_distribution_heatmap.png")

# Define CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_data, validation_data=test_data, epochs=10)
model.save("cnn_model_from_images.h5")

# Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_over_epochs.png")

# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_over_epochs.png")

# Confusion Matrix
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation="vertical", colorbar=False)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# Confidence Histogram
confidences = model.predict(test_data)
max_conf = np.max(confidences, axis=1)
plt.figure(figsize=(6, 4))
plt.hist(max_conf, bins=20, color="purple", edgecolor="black")
plt.title("Histogram of Prediction Confidence")
plt.xlabel("Confidence Score")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("prediction_confidence_histogram.png")
