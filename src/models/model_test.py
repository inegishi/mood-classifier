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

#model output: number
num_to_mood = {0: "sadness",
               1:"joy",
               2: "love",
               3: "anger",
            #    4: "fear",
            #    5: "suprise"
            }

model = joblib.load("model/mood_classifier_kaggle.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

test = [
    "I can't stop crying, it just hurts so much.",  # sadness
    "I’m so proud of you! You did it!! 🎉",         # joy
    "Why would you lie to me like that?",           # anger
    "Every day with you feels like a blessing.",    # love
    "I feel completely alone even in a crowd.",     # sadness
    "This is the best day of my life!",             # joy
    "I’m seriously done with all this nonsense.",   # anger
    "I love the way you laugh, it makes me smile.", # love
    "You promised you'd be there and you weren't.", # sadness
    "Guess what? I got the job!! 😄",               # joy
    "You always do this, you never listen!",        # anger
    "Even after all this time, I still love you.",  # love
    "I wish things were different…",                # sadness
    "I can’t believe how happy I am right now.",    # joy
    "Don’t talk to me like I’m stupid.",            # anger
    "You mean the world to me.",                    # love
]


X = vectorizer.transform(test)

y_pred = model.predict(X)
for i in y_pred:
    print(num_to_mood[i])