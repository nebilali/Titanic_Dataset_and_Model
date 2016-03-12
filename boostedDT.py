'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
import math
from sklearn import tree

from sklearn.metrics import accuracy_score

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.nBoostIter = numBoostingIters
        self.depth = maxTreeDepth
        self.Betas = []
        self.classifiers = []
        self.K = 0
        self.labels = []
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        
        self.labels = list(set(y))
        K = len(self.labels)
        self.K = K
        n,d = X.shape
        w = np.array([1.0/n] * n )
        classifiers = []
        Betas = []
        
        for i in range(self.nBoostIter): 
            clf = tree.DecisionTreeClassifier(max_depth=self.depth)
            clf = clf.fit(X,y,sample_weight=w)
            y_pred = clf.predict(X)
            
            #e = (y_pred!=y).dot(w) / sum(w)
            e = sum(w[y_pred!=y]) / sum(w)

            B = float( (np.log((1-e)/(e)) + np.log(K-1)) ) 
            
            w = np.exp(B*(y!=y_pred)) * w

            w = w / sum(w)
            classifiers.append(clf)
            Betas.append(B)

        self.Betas = Betas
        self.classifiers = classifiers
     
    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        clf = self.classifiers 
        Betas = np.array(self.Betas)
        n,d = X.shape
        K = self.K

        predictions = np.zeros(n)
        predBeta = [0]*K
        preds = np.zeros([n,self.nBoostIter])
        for i in range(self.nBoostIter):
            preds[:,i] = clf[i].predict(X)
            
        for i in range(n): 
            for j in range(K):
                predBeta[j] = ( preds[i,:] == self.labels[j] ).dot(Betas)
            predictions[i] = self.labels[predBeta.index(max(predBeta))]
        
        return predictions






