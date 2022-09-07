import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('/home/arnav/Documents/tsf/task1/student_scores.csv')

x = df[['Hours']]
y = df[['Scores']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)

lr = LinearRegression()
lr.fit(x_train, y_train)
print('[INFO] Model is training...')

pred = lr.predict(x_test)
print(f'[INFO] Model Explainability: {r2_score(y_test, pred)*100:.2f} %')

print("[INFO] Model ready for prediction... \n")

# objectives check
answer = lr.predict([[9.5]])
print(f"If a student studies for 9.5hrs/day, then his/her score will be: {answer[0][0]:.2f}")
