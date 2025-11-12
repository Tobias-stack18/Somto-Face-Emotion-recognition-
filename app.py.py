import os
import sqlite3
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime

# ------------------------------
# ✅ App configuration
# ------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------------------
# ✅ Load model
# ------------------------------
MODEL_PATH = "face_emotionModel.h5"
model = load_model(MODEL_PATH)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ------------------------------
# ✅ Database setup
# ------------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    image_path TEXT,
                    emotion TEXT,
                    date_uploaded TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# ✅ Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user info
        name = request.form.get("name")
        email = request.form.get("email")
        file = request.files["image"]

        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        preds = model.predict(face)[0]
        emotion = EMOTIONS[np.argmax(preds)]

        # Save to DB
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, image_path, emotion, date_uploaded) VALUES (?, ?, ?, ?, ?)",
                  (name, email, file_path, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        # Response messages
        messages = {
            "Happy": "You are smiling. Why are you happy?",
            "Sad": "You look sad. Everything okay?",
            "Angry": "You seem angry. Take a deep breath.",
            "Surprise": "You look surprised! What happened?",
            "Fear": "You seem scared. Don’t worry, you’re safe here.",
            "Disgust": "You look disgusted. Yikes!",
            "Neutral": "You look calm and neutral."
        }
        return jsonify({
            "emotion": emotion,
            "message": messages.get(emotion, "Emotion detected.")
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed."}), 500

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
