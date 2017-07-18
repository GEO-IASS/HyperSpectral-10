# %matplotlib inline

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import dataset
from sklearn.utils.multiclass import check_classification_targets, type_of_target


'''Data loading and initial manipulation'''
df = pd.read_csv('C:\\Users\Michael Byrd\Documents\StapletonLab\headers3mgperml.csv')

x = df.iloc[:, 15:]

y_cat, y_num = pd.get_dummies(pd.DataFrame(df.iloc[:, 1:5])), df.iloc[:, 5:14].drop('PKID', 1, inplace=False)

y = pd.concat((y_cat, y_num), axis=1)

# print(y_cat)

# # change to numpy array
# dataset = dataset.values
#
# # split Features from Responses
# x = dataset[:, 15:]
# y_cat = dataset[:, 1:5]
# y_num = np.concatenate((dataset[:, 5:8], dataset[:, 9:14]), axis=1)
# y = np.concatenate((y_cat, y_num), axis=1)
#
# y_copy = np.copy(y)


# print(y_copy)

def stringToInt():
    dataDict = [{}, {}, {}, {}]
    for j in range(1, 5):
        index = 0
        for i in range(len(df)):
            if df[i][j] not in dataDict[j - 1].keys():
                dataDict[j - 1][df[i][j]] = index
                index += 1
    return dataDict


# dataDict = stringToInt()


def singleResponse(Dict):
    genotype = df[:, 1:2].ravel()
    genotypeDict = Dict[0]
    for i in range(len(genotype)):
        genotype[i] = genotypeDict[genotype[i]]
    density = df[:, 2:3].ravel()
    densityDict = Dict[1]
    for i in range(len(density)):
        density[i] = densityDict[density[i]]
    nitrogen = df[:, 3:4].ravel()
    nitrogenDict = Dict[2]
    for i in range(len(nitrogen)):
        nitrogen[i] = nitrogenDict[nitrogen[i]]
    hormone = df[:, 4:5].ravel()
    hormoneDict = Dict[3]
    for i in range(len(hormone)):
        hormone[i] = hormoneDict[hormone[i]]
    return genotype, density, nitrogen, hormone


# genotype, density, nitrogen, hormone = singleResponse(dataDict)

# integerData = [genotype, density, nitrogen, hormone]

# for i in range(4):
#     for j in range(len(y_copy)):
#         y_copy[j][i] = integerData[i][j]


def logReg(response):
    X_train, X_test, y_train, y_test = train_test_split(x, response, test_size=0.4)  # , random_state=4)
    print(len(X_train), len(X_test))
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))


def ridgeReg(features, response):
    X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.4)  # , random_state=4)
    ridgeR = Ridge()
    ridgeR.fit(X_train, y_train)
    print(ridgeR.score(X_test, y_test))


def ridgeClass(features, response):
    from sklearn.linear_model import RidgeClassifier
    X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.4)  # , random_state=4)
    ridgeC = RidgeClassifier()
    ridgeC.fit(X_train, y_train)
    print(ridgeC.score(X_test, y_test))


# Taken away hormone
# y_copy = np.concatenate((y_copy[:, 0:3], y_copy[:, 4:]), axis=1)

y_numVal = y_num.values

for i in range():
    X_train, X_test, y_train, y_test = train_test_split(x, y_num, test_size=0.4)



# knn = KNeighborsClassifier(n_neighbors=1)
#
# # print('{} \n {}'.format(x.shape, genotype.shape))
# knn.fit(X_train, y_train)
#
# predicted = knn.predict(X_test)
#
#
# y_testVal = y_test
#
#
# falseCount = 0
#
# # for i in range(len(predicted)):
# #     for j in range(len(predicted[0])):
# #         if predicted[i][j] != y_testVal[i][j]:
# #             falseCount += 1
#
# # print(100 - ((falseCount / (X_test.shape[0] * X_test.shape[1]))*100))
#
# print(accuracy_score(y_test, predicted))

