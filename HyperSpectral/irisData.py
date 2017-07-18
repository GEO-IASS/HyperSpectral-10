import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# import iris dataset
iris = datasets.load_iris()
irisData = iris.data
irisFeatures = iris.feature_names
irisTargets = iris.target
irisTargetNames = iris.target_names

X = irisData
y = irisTargets

