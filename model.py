import cv2
import face_recognition
import os
import numpy as np
import pickle

# Path to dataset
DATASET_PATH = "dataset/"
MODEL_PATH = "trained_model/face_encodings.pkl"

# Load dataset and train the model
def train_model():
    known_encodings = []
    known_names = []

    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_image)
            if len(face_locations) > 0:
                encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save encodings to file
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
