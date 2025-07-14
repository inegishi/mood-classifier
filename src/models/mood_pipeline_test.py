import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
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

#inital practice data
data = {
    "message": [
        "i'm so glad you called!",
        "ugh you seriously forgot again?",
        "i don't feel like doing anything today",
        "wait WHAT? no way 😳",
        "thank you for the flowers, they're beautiful!",
        "you left me on read again...",
        "i miss how things used to be",
        "omg i got the job!!!",
        "today was so chill and fun",
        "why didn7t you say anything last night?",
        "everything just feels so heavy right now",
        "you're not gonna believe what just happened!",
        "your little note made my day ☀️",
        "i can't believe you'd say that",
        "i wish i could just disappear sometimes",
        "you seriously got me tickets??!",
        "spending time with you always cheers me up",
        "stop acting like nothing happened",
        "i feel so alone these days",
        "how did you even pull that off?!"
    ],
    "mood": [
        "happy", "mad", "sad", "surprise", 
        "happy", "mad", "sad", "surprise", 
        "happy", "mad", "sad", "surprise", 
        "happy", "mad", "sad", "surprise", 
        "happy", "mad", "sad", "surprise"
    ]
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"]
)
X_df = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
print(X_df.head())

encoder = LabelEncoder()
y_df = encoder.fit_transform(df["mood"])

model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, shuffle= True, random_state=42)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

comparison = pd.DataFrame({
    "actual": y_test,
    "predicted": prediction
}
)

print(comparison.head(50))
