import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load dataset

df = pd.read_csv('C:\\Users\Michael Byrd\Documents\StapletonLab\headers3mgperml.csv')
dfValues = df.values

# Observations
X = df.iloc[:, 15:]

# Responses Split
y_cat, y_num = pd.get_dummies(pd.DataFrame(df.iloc[:, 1:5])), df.iloc[:, 5:13].drop('PKID', 1, inplace=False)

lipids = y_num.iloc[:, 3:]

# Responses
y = pd.concat((y_cat, y_num), axis=1)

# KNN
# X_train, X_test, y_train, y_test = train_test_split(x, y_num)
# for i in range(1, 11):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     print(metrics.accuracy_score(y_test, y_pred), i)


for col in lipids.columns:
    for index, row in lipids.iterrows():
        lipids.set_value(index, col, int(lipids[col][index] * (10 ** 6)))



X_train, X_test, y_train, y_test = train_test_split(X, lipids)
forest = ExtraTreesClassifier()
forest.fit(X_train, y_train)


