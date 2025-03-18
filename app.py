import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
import datetime
import os
import base64
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
from urllib.parse import quote as url_quote  # FIXED IMPORT

app = Flask(__name__)

# Load trained model
MODEL_PATH = "trained_model/face_encodings.pkl"
with open(MODEL_PATH, "rb") as f:
    known_encodings, known_names = pickle.load(f)

attendance_file = "attendance.csv"

# Initialize attendance file
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    # Decode base64 image
    image_data = image_data.split(",")[1]  # Remove header
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)

    # Convert RGB to BGR (for OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process face recognition
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    recognized_names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            best_match = np.argmin(face_recognition.face_distance(known_encodings, encoding))
            name = known_names[best_match]

            # Log attendance
            df = pd.read_csv(attendance_file)
            df = pd.concat([df, pd.DataFrame([{"Name": name, "Time": datetime.datetime.now()}])], ignore_index=True)
            df.to_csv(attendance_file, index=False)

        recognized_names.append(name)

    return jsonify({"recognized": recognized_names})

if __name__ == "__main__":
    app.run(debug=True)