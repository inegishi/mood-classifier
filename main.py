import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import pytesseract as pyt
import cv2 as cv

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_raw_text(file_path):
    ignore = ["Messages", "Details"]

    img = cv.imread(file_path)
    text = pyt.image_to_string(img)

    messages = [
        msg.replace("\n", " ").strip()
        for msg in text.split("\n\n")
        if msg.strip() and len(msg) >= 3 and msg.strip() not in ignore
    ]

    return messages



num_to_mood = {0: "sadness  😢",
               1:"joy 😊" ,
               2: "anger 😠" ,
            #    3: "love",
            #    4: "fear",
            #    5: "suprise"
            }


model = joblib.load("model/mood_classifier_kaggle.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


data_path = "dataset/test.jpg"
text = extract_raw_text(data_path)

X = vectorizer.transform(text)
y_pred = model.predict(X)

for i in range(len(y_pred)):
   print(f"message {i+1}: {text[i]} \nmood: {num_to_mood[y_pred[i]]}\n")