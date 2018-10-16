# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy as sp
import random as rn
import matplotlib.pyplot as plt
import pylab
from time import time
from mpl_toolkits.mplot3d import Axes3D

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

from sklearn import manifold, datasets
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE 
from sklearn.cluster import KMeans
from matplotlib.ticker import NullFormatter

import xgboost as xgb
from xgboost import XGBClassifier

def mnist(eval_1hot):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=eval_1hot)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    return x_train, y_train, x_test, y_test

def pca(n,x):
    print('Running PCA') 
    t0 = time()
    pca = PCA(n_components=n, svd_solver='randomized', whiten=True)
    x_pca = pca.fit_transform(x)
    x_proj_pca = pca.inverse_transform(x_pca)
    print("done in %0.3fs" % (time() - t0))
    print("(see results below)")
    return x_pca, x_proj_pca

def kpca(n,x,kernel):
    print('Running KPCA') 
    t0 = time()
    kpca = KernelPCA(n_components=n, kernel=kernel, fit_inverse_transform=True)#, gamma=10)
    x_kpca = kpca.fit_transform(x)
    x_proj_kpca = kpca.inverse_transform(x_kpca)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_kpca, x_proj_kpca

def lda(n,x,y):
    print('Running LDA') 
    t0 = time()
    x_lda = LinearDiscriminantAnalysis(n_components=n, store_covariance=False).fit_transform(x, y)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_lda

def isomap(n,x):
    print('Running ISOMAP') 
    t0 = time()
    x_isomap = Isomap(n_components=n,n_neighbors=30).fit_transform(x)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_isomap

def lle(n,nay,x):
    print('Running LLE') 
    t0 = time()
    x_lle = LocallyLinearEmbedding(n_components=n, n_neighbors=nay).fit_transform(x)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_lle

def mds(n,x):
    print('Running MDS') 
    t0 = time()
    x_mds = MDS(n_components=n).fit_transform(x)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_mds

def tsne(n,x,perp):
    print('Running TSNE with Perplexity:', perp) 
    t0 = time()
    x_tsne = TSNE(n_components=n, perplexity=perp).fit_transform(x)
    print("done in %0.3fs" % (time() - t0))  
    print("(see results below)")
    return x_tsne

def plotz(x,x_proj,N):
    org_pixels = x.reshape((N, N))
    mod_pixels = x_proj.reshape((N, N))
    #plt.title('Label is {label}'.format(label=label))
    plt.title('Before')
    plt.imshow(org_pixels, cmap='gray')
    plt.show()
    plt.title('After')
    plt.imshow(mod_pixels, cmap='gray')
    plt.show()
    
def two_dim_plotz(x,y):
    scat= plt.scatter(x[:,0], x[:,1], c=y, cmap=pylab.cm.gist_rainbow)
    cb = plt.colorbar(scat, spacing='proportional')
    cb.set_label('Label')
    plt.show()

#PreProcessing
scaler = StandardScaler()
normz = Normalizer()
x_train, y_train, x_test, y_test = mnist(False)
x_train_scaled = scaler.fit_transform(x_train)
x_train_normzd = normz.fit_transform(x_train)

#T1
print('Task 1 --->>') 
dim=[1,10,20,50,100,200]
for i in dim:
    print('\n\n DIMENSION:',i)
    print('\nWithout Normalization, Dimension:', i)
    x_train_pca, x_train_proj_pca = pca(i,x_train)
    print ('MSE', mean_squared_error(x_train, x_train_proj_pca))
    plotz(x_train[0],x_train_proj_pca[0],28)
    
    print('\nWith Normalization, Dimension:', i)
    x_train_pca, x_train_proj_pca = pca(i,x_train_normzd)
    print ('MSE', mean_squared_error(x_train_normzd, x_train_proj_pca))
    plotz(x_train[0],x_train_proj_pca[0],28)
    
    print('\nStandard Scaling, Dimension:', i)
    x_train_pca, x_train_proj_pca = pca(i,x_train_scaled)
    print ('MSE', mean_squared_error(x_train_scaled, x_train_proj_pca))
    plotz(x_train[0],x_train_proj_pca[0],28)
    
#T2
dim=2
print('\n\nTask 2 --->>')

#pca
print('\nWith Normalization, Plot Dimension:', dim)
x_train_pca, x_train_proj_pca = pca(2,x_train_normzd)
two_dim_plotz(x_train_pca, y_train)

#kpca
print('\nWith Normalization (on 10k obs only), Plot in Dimension:', dim)
x_train_kpca, x_train_proj_kpca = kpca(2,x_train_normzd[0:10000],"rbf")
two_dim_plotz(x_train_kpca, y_train[0:10000])

#lda
print('\nWith Normalization, Plot in Dimension:', dim)
x_train_lda = lda(2,x_train_normzd,y_train)
two_dim_plotz(x_train_lda, y_train)

#isomap
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
x_train_isomap = isomap(2,x_train_normzd[0:1000])
two_dim_plotz(x_train_isomap, y_train[0:1000])

#lle
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
x_train_lle = lle(2,30,x_train_normzd[0:1000])
two_dim_plotz(x_train_lle, y_train[0:1000])

#mds
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
x_train_mds = mds(2,x_train_normzd[0:1000])
two_dim_plotz(x_train_mds, y_train[0:1000])

#TSNE
for p in [10,30,50]:
    print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim, ', Perp=', p)
    x_train_tsne = tsne(2,x_train_normzd[0:1000],p)
    two_dim_plotz(x_train_tsne, y_train[0:1000])

#T3    
print('\n\nTask 3 --->>')
Axes3D    
n_points = 1000
X, color = datasets.samples_generator.make_s_curve (n_points, random_state = 0)
n_neighbors = 10
n_components = 2
fig = plt.figure(figsize=(50, 30))
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.show()  

dim=2
#pca
print('\nWith Normalization, Plot Dimension:', dim)
X_pca, X_proj_pca = pca(2,X)
two_dim_plotz(X_pca, color)

#determine y for lda
v = []
for i in range(1, 20):
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    km.fit(X)
    v.append(km.inertia_)
plt.plot(range(1, 20), v)
plt.title('Determine Ideal Cluster Size')
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
Y = kmeans.fit_predict(X)

#lda
print('\nWith Normalization, Plot in Dimension:', dim)
x_train_lda = lda(2,X,Y)
two_dim_plotz(X, Y)

#isomap
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
X_isomap = isomap(2,X)
two_dim_plotz(X_isomap, color)

#lle
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
X_lle = lle(2,30,X)
two_dim_plotz(X_lle, color)

#mds
print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim)
X_mds = mds(2,X)
two_dim_plotz(X_mds, color)

#TSNE
for p in [10,30,50]:
    print('\nWith Normalization (on 1k obs only), Plot in Dimension:', dim, ', Perp=', p)
    X_tsne = tsne(2,X,p)
    two_dim_plotz(X_tsne, color)

#T4
print('\n\nTask 4 --->>')
iris = datasets.load_iris()
x = iris.data
y = iris.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

#less memory
dump_svmlight_file(xtrain, ytrain, 'dtrain.svm', zero_based=True)
dump_svmlight_file(xtest, ytest, 'dtest.svm', zero_based=True)
xtrain_svm = xgb.DMatrix('dtrain.svm')
xtest_svm = xgb.DMatrix('dtest.svm')
param = {'max_depth':3, 'n_estimators':100, 'eta':0.3, 'objective':'multi:softprob', 'num_class':3}
bst = xgb.train(param, xtrain_svm, 1)
#bst.dump_model('dump.raw.txt')
ytrain_prob = bst.predict(xtrain_svm)
outtrain = np.asarray([np.argmax(i) for i in ytrain_prob])
ytest_prob = bst.predict(xtest_svm)
outtest = np.asarray([np.argmax(i) for i in ytest_prob])
accuracy = accuracy_score(outtrain, ytrain)
print("Model performance based on specified params")
print("Train Accuracy: %.2f%%" % (accuracy * 100.0))
accuracy = accuracy_score(outtest, ytest)
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

#grid search
print("\nImplementing Grid Search")
model = XGBClassifier()
param_grid = {'max_depth':range(3,10), 'n_estimators':range(50, 500, 50)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x, y)
# summarize results
print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']

    
