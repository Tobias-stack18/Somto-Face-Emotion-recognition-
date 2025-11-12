import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ------------------------------
# Load dataset
# ------------------------------
data = pd.read_csv("dataset/fer2013.csv")

pixels = data["pixels"].tolist()
faces = [np.fromstring(p, dtype=int, sep=" ").reshape(48, 48, 1) for p in pixels]
faces = np.array(faces, dtype="float32") / 255.0
emotions = tf.keras.utils.to_categorical(data["emotion"], num_classes=7)

x_train, x_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# ------------------------------
# Build CNN model
# ------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ------------------------------
# Train & save model
# ------------------------------
checkpoint = ModelCheckpoint("face_emotionModel.h5", save_best_only=True, monitor="val_accuracy", mode="max")
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, callbacks=[checkpoint])

print("✅ Model training complete — Saved as face_emotionModel.h5")
