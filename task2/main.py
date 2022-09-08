import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

x = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

model = DecisionTreeClassifier()
print('\n[INFO] Model trained...')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Model is {accuracy_score(y_test, pred)*100:.2f} % accurate")
print("\n[INFO] Decision Tree Visualizer saved...")
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=x.columns,  
                   class_names=list(set(y)),
                   filled=True)
fig.savefig("dtc_visualize.png")

print('\n[INFO] Ready for prediction...')

# user input
sepalL = float(input("Enter the Sepal Length (cm): "))
sepalW = float(input("Enter the Sepal Width (cm): "))
petalL = float(input("Enter the Petal Length (cm): "))
petalW = float(input("Enter the Petal Width (cm): "))

data = [[sepalL, sepalW, petalL, petalW]]
final_pred = model.predict(data)[0]
print(f"The predicted species is: '{final_pred}'")
