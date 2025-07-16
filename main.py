import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from datetime import datetime

#from brokenaxes import brokenaxes
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings 
warnings.filterwarnings('ignore')

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


data_path = "dataset/test5.png"
text = extract_raw_text(data_path)

X = vectorizer.transform(text)
y_pred = model.predict(X)

for i in range(len(y_pred)):
   print(f"message {i+1}: {text[i]} \nmood: {num_to_mood[y_pred[i]]}\n")