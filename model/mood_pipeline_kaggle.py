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

df = pd.read_csv("dataset/text.csv") 
#shape: (416809,3). #index, sentence, label.
#LABEL: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

encoder = LabelEncoder()
y_df = encoder.fit_transform(df["label"])

model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.2, shuffle= True, random_state=42)
model.fit(X_train,y_train)

joblib.dump(model,"model/mood_classifier_kaggle.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

prediction = model.predict(X_test)
comparison = pd.DataFrame({
    "actual": y_test,
    "predicted": prediction
}
)

print(comparison.head(50))



