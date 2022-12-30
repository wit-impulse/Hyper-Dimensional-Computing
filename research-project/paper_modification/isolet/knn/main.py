# -*- coding: utf-8 -*-

# knn for predicting eye state
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np


# Measurements script can be called here according to needs

with open('/Users/austinvas/Documents/mywork/isolet/dataset/isolet.pkl', 'rb') as f:
    isolet = pickle.load(f)
X_train, y_train, X_test, y_test = isolet


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

best_param1 = {'n':8 ,'metric':'minkowski' , 'p':2, 'acc':91.59 }
best_param2 = {'n':9 ,'metric':'minkowski' , 'p':2, 'acc':92.11 }


# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors = 9, metric='minkowski', p=2, n_jobs= -1 )
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
acc=accuracy_score(y_test, y_pred)
print(acc)