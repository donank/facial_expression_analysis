#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

training_data = pd.read_csv("training-part-2.csv")
test_data = pd.read_csv("test-part-2.csv")


class NB:
    def hot_encoding(self, y):
        for i in range(len(y)):
            if y[i] == "smile":
                y[i] = 1
            else:
                y[i] = 0
        return y

    def fit(self, x, y, K):
        n, d = x.shape #number of data elements/rows
        self.K = K #number of classes
        self.props = np.zeros([self.K,d]) 
        self.classes = np.zeros([self.K])
        y = self.hot_encoding(y)
        for k in range(self.K):
            X_k = x[y == k]
            self.props[k] = np.mean(X_k, axis = 0) 
            self.classes[k] = X_k.shape[0]/float(n)
        

    def predict(self, x):
        n, d = x.shape
        x = np.reshape(x, (1, n, d))
        self.props = np.reshape(self.props, (self.K, 1, d))
        #prevent underflow
        self.props = self.props.clip(1e-12, 1-1e-12)
        #compute probabilities
        log_py = np.log(np.tile(self.classes.reshape((self.K, 1)), (1,n)).reshape([self.K, n, 1]))
        log_pxy = x * np.log(self.props) + (1-x) * np.log(1-self.props)
        log_pyx = (log_pxy + log_py).sum(axis=2)

        return log_pyx.argmax(axis=0).flatten()

X_train = training_data.iloc[:, 0:17].values
y_train = training_data.iloc[:, 17].values

X_test = test_data.iloc[:, 0:17].values
y_test = test_data.iloc[:, 17].values

#nb = NB()
#nb.fit(X_train, y_train, 2) #Fit the training data
#pred_test = nb.predict(X_train) #Performing Prediction

clf = QDA()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(pred)

'''
for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = "frown"
    else:
        y_train[i] = "smile"
        
#Converting predicted output in the form of class values (0/1 to frown/smile)
pred_test_var = []
for i in range(len(pred_test)):
    if pred_test[i] == 0:
        pred_test_var.append("frown")
    else:
        pred_test_var.append("smile")
    
pred_df = pd.DataFrame()
pred_df["Predicted"] = pred_test_var
pred_df["Actual"] = y_train

print(pred_df)

y_test = nb.hot_encoding(y_train)
'''
pred_sum = 0
for i in range(len(y_test)):
    if y_test[i] == pred[i]:
        pred_sum += 1
    
err_percent = 1 - (pred_sum/len(y_test))
print("Error Rate:", err_percent*100)

