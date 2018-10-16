# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:11:57 2018

@author: nc57
"""
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
N, D = X.shape
trainsize = 105
testsize = 45
np.random.seed(0)
index = np.random.permutation(N)
indextrain = index[0:trainsize]
indextest = index[trainsize : : ]
Xtrain = X[indextrain,:]
ytrain = y[indextrain ]
Xtest = X[indextest,:]
ytest = y[indextest]
Dataset=np.append(Xtrain,ytrain.reshape(-1,1),axis=1)

#test split
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
#gini index
def gini_index(groups, classes):
	n_instances = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
#select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
#create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
#create child splits for node or terminate
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	#check for no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	#check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	#process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	#process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
#Training decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
#view decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
 
#tree parameters
max_depth = 5
min_size = 20
print('\nInitializaing Parameters..')
print('Max depth', max_depth, 'Min size', min_size)
print('\nTraining..')
tree = build_tree(Dataset, max_depth, min_size)
print('\nPrinting Tree..')
print_tree(tree)

#function for predicting output
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
datatest =np.append(Xtest,ytest.reshape(-1,1),axis=1)
datatrain =np.append(Xtrain,ytrain.reshape(-1,1),axis=1)

#predict with the tree
print('\nTesting tree on TRAIN dataset..')
output_train = []
for row in datatrain:
    output_train.append(predict(tree, row))
train_accuracy = float((ytrain == np.array(output_train)).sum())/float(ytrain.shape[0])
print ('Train acuracy', train_accuracy)

print('\nTesting tree on TEST dataset..')
output_test = []
for row in datatest:
    output_test.append(predict(tree, row))
test_accuracy = float((ytest == np.array(output_test)).sum())/float(ytest.shape[0])
print ('Test acuracy', test_accuracy)
