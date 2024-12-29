from pyexpat import model
import pandas as pd
import yfinance as yf
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


iris = load_iris()
x = iris.data
y = iris.target


model = KNeighborsClassifier(n_neighbors=6)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)
# k= np.ndarray()
print(cross_val_score(model , x_train,y_train,scoring="accuracy",cv=5))
# model.fit(x_train,y_train)
# print(model.score(x_test,y_test))
# print(x)