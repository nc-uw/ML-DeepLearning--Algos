# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:18:52 2018

@author: nc57
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM

def datacreation(means,covariance):
    dataset=[]
    for i in range(0,1000):
        x=np.random.multivariate_normal(means[np.random.randint(0,len(means))],covariance)
        dataset.append(x)
    return dataset 
        
def calculate_first_moment(X):
    mu = np.zeros((5, 1))
    for t in range(len(X)):
        for i in range(len(X[t])):
            mu[i] +=  X[t][i]
    mu /= len(X)
    return mu

def calculate_second_moment(X):
    Sigma = np.zeros((len(X[0]), len(X[0])))
    for t in range(len(X)):
        for i in range(len(X[t])):
            for j in range(len(X[t])):
                Sigma[i][j] += np.dot(X[t][i],X[t][j])
    Sigma /= len(X)
    return Sigma

def extract_information_from_second_moment(Sigma, X):
    U, S, _ = np.linalg.svd(Sigma)
    s_est = S[-1]
    W, X_whit = perform_whitening(X, U, S)
    return (s_est, W, X_whit)

def perform_whitening(X, U, S):
    W = np.matmul(U[:, 0:components], np.sqrt(np.linalg.pinv(np.diag(S[0:components]))))
    X_whit = np.matmul(X, W)
    return (W, X_whit)

def perform_tensor_power_method(X_whit, W, s_est, mu):
    TOL = 1e-8
    maxiter = 100
    V_est = np.zeros((components, components))
    lamb = np.zeros((components, 1))

    for i in range(components):
        v_old = np.random.rand(components, 1)
        v_old = np.divide(v_old, np.linalg.norm(v_old))
        for iter in range(maxiter):
            v_new = (np.matmul(np.transpose(X_whit), (np.matmul(X_whit, v_old) * np.matmul(X_whit, v_old)))) / len(X_whit)
            #v_new = v_new - s_est * (W' * mu * dot((W*v_old),(W*v_old)));
            #v_new = v_new - s_est * (2 * W' * W * v_old * ((W'*mu)' * (v_old)));
            v_new -= s_est * (np.matmul(np.matmul(W.T, mu), np.dot(np.matmul(W, v_old).T,np.matmul(W, v_old))))
            v_new -= s_est * (2 * np.matmul(W.T, np.matmul(W, np.matmul(v_old, np.matmul(np.matmul(W.T, mu).T, v_old)))))
            if i > 0:
                for j in range(i):
                    v_new -= np.reshape(V_est[:, j] * np.power(np.matmul(np.transpose(v_old), V_est[:, j]), 2) * lamb[j], (components, 1))
            l = np.linalg.norm(v_new)
            v_new = np.divide(v_new, np.linalg.norm(v_new))
            if np.linalg.norm(v_old - v_new) < TOL:
                V_est[:, i] = np.reshape(v_new, components)
                lamb[i] = l
                break
            v_old = v_new
    
    return (V_est, lamb)

#create data
cov1=0.25*np.identity(5)
cov2=np.identity(5)
Kd=[[-0.5,1.1,0.2,-0.9,0.2],[0.2,-0.1,0.5,-0.8,1.0],[-0.3,0.2,0.9,0.7,1.0],[0.2,0.9,0.1,-0.4,0.5]]
dataset1=datacreation(Kd,cov1)
dataset2=datacreation(Kd,cov2)
components=4

#EM algo
print('.. Running EM algorithm..  ')
gmm1 = GMM(n_components=4)
gmm1.fit(dataset1)
gmm2 = GMM(n_components=4)
gmm2.fit(dataset2)

print(' Predicted means of 1st mixture = \n',gmm1.means_)
print(' Predicted scovariance of 1st mixture = \n',gmm1.covars_)
print('\n')
print(' Predicted means of 2nd mixture = \n',gmm2.means_)
print(' Predicted scovariance of 2nd mixture = \n',gmm2.covars_)