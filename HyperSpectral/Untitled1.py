
# coding: utf-8

# In[155]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[156]:

# import dataset
dataset = pd.read_csv('C:\\Users\Michael Byrd\Documents\StapletonLab\headers3mgperml.csv')

# change to numpy array
dataset = dataset.as_matrix()


# In[157]:

# split Features from Responses
x = dataset[:, 15:]
y_cat = dataset[:, 1:5]
y_num = np.concatenate((dataset[:, 5:8], dataset[:, 9:14]), axis=1)
y = np.concatenate((y_cat, y_num), axis=1)


# In[158]:

# One Response
hormone = dataset[:, 4:5]
# print(hormone)
# Changes to a 1D array
hormone = np.ravel(genotype)
print(hormone)


# In[159]:

dataDict = [{}, {}, {}, {}]
for j in range(1, 5):
    index = 0
    for i in range(len(dataset)):
        if dataset[i][j] not in dataDict[j - 1].keys():
            dataDict[j - 1][dataset[i][j]] = index
            index += 1
dataDict


# In[160]:

genotype = dataset[:, 1:2].ravel()
genotypeDict = dataDict[0]
for i in range(len(genotype)):
    genotype[i] = genotypeDict[genotype[i]]
    
density = dataset[:, 2:3].ravel()
densityDict = dataDict[1]
for i in range(len(density)):
    density[i] = densityDict[density[i]]
    
nitrogen = dataset[:, 3:4].ravel()
nitrogenDict = dataDict[2]
for i in range(len(nitrogen)):
    nitrogen[i] = nitrogenDict[nitrogen[i]]
    
hormone = dataset[:, 4:5].ravel()
hormoneDict = dataDict[3]
for i in range(len(hormone)):
    hormone[i] = hormoneDict[hormone[i]]



# In[161]:

X_train, X_test, y_train, y_test = train_test_split(x, hormone, test_size=0.4)

# print('{} \n {}'.format(y_train, y_test))

ridgeR = Ridge()

ridgeR.fit(X_train, y_train)

y_pred = ridgeR.predict(X_test)

print(ridgeR.score(X_test, y_test))


# In[162]:

from sklearn import linear_model

kernelWT = dataset[:, 5:6]

kernelWT = kernelWT.transpose()[0]

# print(kernelWT)


X_train, X_test, y_train, y_test = train_test_split(x, kernelWT, test_size=0.4)

# print('{} \n {}'.format(y_train, y_test))

ridgeR = linear_model.BayesianRidge()

ridgeR.fit(X_train, y_train)

y_pred = ridgeR.predict(X_test)

print(ridgeR.score(X_test, y_test))


# In[ ]:



