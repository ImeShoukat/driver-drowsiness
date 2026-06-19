import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("OpenCV Version:", cv2.__version__)
print("TensorFlow Version:", tf.__version__)

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clah_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clah_img.apply(gray_img)
    final_img = cv2.resize(enhanced_img, (64, 64))
    return final_img/255.0

import matplotlib.pyplot as plt
sample_path = "data/train/awake/s0001_01842_0_0_1_0_0_01.png"
if os.path.exists(sample_path):
    img = cv2.imread(sample_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_image = preprocess_image(sample_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()

    plt.show()
else:
    print(f"sampel tidak ditemukan {sample_path}. cek dan coba lagi.")

def load_data(data_dir):
    X, y = [], []
    
    tags = {'awake': 1, 'sleepy': 0}
    for tag_name, label in tags.items():
        path_folder = os.path.join(data_dir, tag_name)
        if not os.path.exists(path_folder):
            print(f"folder {path_folder} tidak ditemukan.")
            continue
        for file_name in os.listdir(path_folder):
            if file_name.endswith('.png'):
                file_path=os.path.join(path_folder, file_name)
                try:
                    img = preprocess_image(file_path)
                    X.append(img)
                    y.append(label)
                except Exception as e:
                    print(f"gagal memproses {file_path}: {e}")
                    continue

    X = np.array(X).reshape(-1, 64, 64, 1)
    y = np.array(y)
    return X, y

X_train, y_train = load_data("data/train")
X_val, y_val = load_data("data/val")

print(f"Data latih: {X_train.shape}, {y_train.shape}")
print(f"Data validasi: {X_val.shape}, {y_val.shape}")

model = models.Sequential([
    #layer 1, kovolusi dan pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),

    #layer 2, filter untuk detail kelompok mata
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    #layer 3, pola spasial
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    #layer 4, klasifikasi
    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    #output
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("training model")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
    shuffle=True
)

print("menyimpan model")

# testing model
X_test, y_test = load_data("data/test")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"akurasi test: {test_acc:.4f}, loss test: {test_loss:.4f}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange', marker='o')
plt.title('Grafik Akurasi Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend(loc='lower right')
plt.grid(True)

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='darkred', marker='o')
plt.title('Grafik Tingkat Eror (Loss) Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()