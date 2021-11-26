#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class GaussianNB:
    def fit(self, X, t):
        self.priors = dict()
        self.means = dict()
        self.covs = dict()
        
        self.classes = np.unique(t)

        for c in self.classes:
            X_c = X[t == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.covs[c] = np.diag(np.diag(np.cov(X_c, rowvar=False)))
            
    def predict(self, X):
        preds = list()
        for x in X:
            posts = list()
            for c in self.classes:
                prior = np.log(self.priors[c])
                inv_cov = np.linalg.inv(self.covs[c])
                inv_cov_det = np.linalg.det(inv_cov)
                diff = x-self.means[c]
                likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)
        return np.array(preds)
        
class QuadraticDiscriminantAnalysis:

    def fit(self, X, y):
        self.labels, self.class_priors = np.unique(y, return_counts=True)
        self.class_priors = self.class_priors / y.shape[0]

        self.Cov = []
        self.Mu = []
        
        for k in range(len(self.labels)):
            X_k = X[y==self.labels[k]]
            self.Mu.append(np.mean(X_k, axis=0))
            self.Cov.append(np.cov(X_k.T))
        
    def predict(self, X):
        labels = []

        for i in range(X.shape[0]):
            labels.append(self.predict_sample(X[i]))
        
        return np.array(labels)

    def predict_sample(self, X):
        max_label = 0
        max_likelihood = 0

        for k in range(len(self.labels)):
            likelihood  = np.exp(-1/2 * (X - self.Mu[k]).T @ np.linalg.inv(self.Cov[k]) @ (X - self.Mu[k]))
            
            if likelihood > max_likelihood:
                max_label = self.labels[k]
                max_likelihood = likelihood
        
        return max_label

class GaussianDiscriminantAnalysis:
    def fit(self, X, t):
        self.priors = dict()
        self.means = dict()
        self.covs = dict()
        
        self.classes = np.unique(t)

        for c in self.classes:
            X_c = X[t == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.covs[c] = np.cov(X_c, rowvar=False)

    def predict(self, X):
        preds = list()
        for x in X:
            posts = list()
            for c in self.classes:
                prior = np.log(self.priors[c])
                inv_cov = np.linalg.inv(self.covs[c])
                inv_cov_det = np.linalg.det(inv_cov)
                diff = x-self.means[c]
                likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)
        return np.array(preds)
