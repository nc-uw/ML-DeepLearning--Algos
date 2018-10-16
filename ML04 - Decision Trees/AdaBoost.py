# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:11:57 2018

@author: nc57
"""

import pandas as pd
import numpy as np
import math
from sklearn import datasets

np.random.seed(42)

bc = datasets.load_breast_cancer()
df = pd.DataFrame(bc.data, columns = bc.feature_names)
df['target'] = 2*pd.Series(bc.target) - 1
df.head()

df_train=df[:469]
df_test=df[469:]
varz = list(df)
varz.remove('target')
dim = len(varz)
y_train = df_train['target']
y_test = df_test['target']
N=y_train.shape[0]
threshold_nos = 100

total_weak_learners = 100 #hyparam

for iters in range(total_weak_learners)[1:]:
    weights = np.ones(N)/N
    weights = np.ones(N)/N
    ada = {}
    bins = {}
    bins = {i:list(pd.cut(df_train[i], threshold_nos, retbins=True)[1]) for i in varz}
    
    #loop - for weaklearner
    ensemble = np.zeros(N)
    for i in range(iters):
    	#check for best stump	
    	best_stump = {}
    	minE=math.inf
    	for var in varz:
    		for thresh in bins[var]:
    			x = df_train[var]
    			pred_gte = np.ones(N)
    			pred_lt = np.ones(N)
    			err_gte = np.ones(N)
    			err_lt = np.ones(N)
    			pred_gte[x >= thresh] = -1.
    			pred_lt[x < thresh] = -1.
    			err_gte[pred_gte == y_train] = 0.
    			err_lt[pred_lt == y_train] = 0.
    			E_gte = np.inner(weights,err_gte)
    			E_lt = np.inner(weights,err_lt)
    			Ei = min(E_gte, E_lt)
    			if E_lt < E_gte:
    				eqi = 'lt'
    				pred_y = pred_lt
    			else:
    				eqi = 'gte'
    				pred_y = pred_gte
    
    			if Ei < minE:
    				minE = Ei
    				best_stump['eq'] = eqi
    				best_stump['var'] = var
    				best_stump['thresh'] = thresh
    				minPred = pred_y
    	#updates
    	alpha = 0.5*math.log((1-minE)/max(minE,1e-10))
    	#print('alpha', alpha)
    	best_stump['alpha'] = alpha
    	delta_weights = np.exp(-1.0*alpha*y_train*minPred)
    	weights = weights * delta_weights
    	weights = weights / weights.sum()
    
    	ensemble += alpha * minPred
    	ensemble_error = np.zeros(N)
    	ensemble_error[np.sign(ensemble) != y_train] = 1
    	error = ensemble_error.sum()/N
    	#print ('best stump', best_stump)
    	#print ('error', error)
    	key='iter'+str(i)
    	ada[key]=best_stump
    	if error == 0: break
    
    pred_y_test = np.zeros(y_test.shape[0])
    for j in ada.keys():
        temp = np.ones(y_test.shape[0])
        if ada[j]['eq'] == 'gte':
            temp[df_test[ada[j]['var']] >= ada[j]['thresh']] = -1
        else:
            temp[df_test[ada[j]['var']] < ada[j]['thresh']] = -1
        pred_y_test += ada[j]['alpha']*temp
    
    test_accuracy = (np.sign(pred_y_test) == y_test).sum()/y_test.shape[0]
    print ('total_weak_learners', iters, 'train accuracy', 1 - error, 'test accuracy', test_accuracy)
    print ('hyper parameter which maximizes train and test accuracy: 22 weak learners')