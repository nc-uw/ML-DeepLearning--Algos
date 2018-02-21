"""
Created on Sat Jan 13 23:41:19 2018

"""
import urllib2 as urel
import numpy as np
import pandas as pd
import random as rnd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#numerical variables
num_vars = ['Lot Area', 'Lot Frontage', 'Year Built',
'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
'2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
'Garage Area', 'Wood Deck SF', 'Open Porch SF',
'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
'Pool Area']

#discrete vars
discr_vars = ['MS SubClass', 'MS Zoning', 'Street',
'Alley', 'Lot Shape', 'Land Contour',
'Utilities', 'Lot Config', 'Land Slope',
'Neighborhood', 'Condition 1', 'Condition 2',
'Bldg Type', 'House Style', 'Overall Qual',
'Overall Cond', 'Roof Style', 'Roof Matl',
'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
'Exter Qual', 'Exter Cond', 'Foundation',
'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
'BsmtFin Type 1', 'Heating', 'Heating QC',
'Central Air', 'Electrical', 'Bsmt Full Bath',
'Bsmt Half Bath', 'Full Bath', 'Half Bath',
'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual',
'TotRms AbvGrd', 'Functional', 'Fireplaces',
'Fireplace Qu', 'Garage Type', 'Garage Cars',
'Garage Qual', 'Garage Cond', 'Paved Drive',
'Pool QC', 'Fence', 'Sale Type', 'Sale Condition']

#read contents on txt file from url
url = 'https://ww2.amstat.org/publications/jse/v19n3/decock/AmesHousing.txt'
data = np.genfromtxt(url, delimiter="\t", dtype=None) #,names=True)
#NOTE 1: by default genfromtxt replaces mssing strings by NA.
#NOTE 2: all the data is read  as string type

#list of headers
head = data[0,:]
#data[1:,:] is the array
data = data[1:,:]

#find position of numerical vars in header list
pos_int = [i for i, x in enumerate(list(head)) if x in num_vars]
#find position of discrete vars in header list
pos_str = [i for i, x in enumerate(list(head)) if x in discr_vars]
#unique values of character variables
data_classes = [set(data[:,i]) for i in pos_str]

#manual 1hot encoding (todo)
##temp = [(discr_vars[i] + ' ' + list(data_classes[i])[j]).replace(" ", "_")]
##1: [1 if data[0,pos_str[i]] == (list(data_classes[i])[j]) else 0
##2: for i in range(len(discr_vars)) for j in range(len(data_classes[i]))]
###todo: create data key + dictionary in spare time!

#1hot encoding via sklearn package (CONVERT TO FUNCTION!)
##integer encode
label_enc = LabelEncoder()
onehot_enc = OneHotEncoder(sparse=False)
int_enc = np.transpose([label_encoder.fit_transform(data[:,pos_str[i]]) for i in range(len(discr_vars))])
##transform encode
data_onehot = onehot_enc.fit_transform(int_enc)
##verify no. of class variables
sum(np.amax(int_enc,0) + 1) == np.shape(data_onehot)[1]

#replace 'NA', '', by '0'
data = data[:,pos_int]
data[data == 'NA'] = '0'
data[data == ''] = '0'
#convert string to int
data = data.astype(float)
#collating all integer types
data = np.append(data, data_onehot, axis = 1)

#split into train test val based on modulus operation on order variable
data_val = data[data[:,0].astype(int) % 5 == 3]
data_test = data[data[:,0].astype(int) % 5 == 4]
data_train = data[data[:,0].astype(int) % 5 != any([3,4])]
