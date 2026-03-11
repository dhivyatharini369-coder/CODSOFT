import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("IMDb Movies India.csv", encoding='latin-1')
print(df.head())
print(df.shape)
df = df.dropna(subset=['Rating'])
df['Genre'] = df['Genre'].fillna("Unknown")
df['Director'] = df['Director'].fillna("Unknown")
df['Actor 1'] = df['Actor 1'].fillna("Unknown")
df = df[['Genre','Director','Actor 1','Votes','Rating']]
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
X = df.drop('Rating', axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
sample = [[10, 250, 1000, 50000]]  # Example encoded values
predicted_rating = model.predict(sample)
print("Predicted Movie Rating:", predicted_rating)