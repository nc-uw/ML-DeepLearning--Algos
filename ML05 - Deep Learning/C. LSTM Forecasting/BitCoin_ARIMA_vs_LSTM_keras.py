import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
import math
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, LeakyReLU, GRU, SimpleRNN

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#fix seed
np.random.seed(77)

#read csv into pd
bt = pd.read_csv('bitcoin_price.csv', header = None, index_col = 0, names = ['price'])
#bt = pd.Series.from_csv('bitcoin_price.csv')
#bt.set_index('month', inplace=True)

#1st order diff for nonstash
bt['price_feature'] = bt['price'].diff()/bt['price']
#dropping nans
bt = bt.dropna()

#pd stats
'''
bt.head()
bt.info()
bt.describe()
'''
bt['price_feature'] = bt['price']

#scale between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(bt['price_feature'].values.reshape(-1,1))

#create datasets as per format
def create_dataset(dataset, ind):
  dataX, dataY = [], []
  indX, indY = [], []
  for i in range(len(dataset)-1):
    dataX.append(dataset[i])
    dataY.append(dataset[i + 1])
    indX.append(ind[i])
    indY.append(ind[i + 1])
  return np.asarray(dataX), np.asarray(dataY), np.asarray(indX), np.asarray(indY)

#data
X, Y, indX, indY = create_dataset(dataset,  bt.index)

#data for arima
trainX = bt['price_feature'][bt['price_feature'].index<'2017-09-01']
testX = bt['price_feature'][bt['price_feature'].index>='2017-09-01']

#ARIMA
print ('\n\n Running Model type: ARIMA')
plot_acf(bt['price_feature'].diff().values[1:], lags=50)
plt.show()
plot_pacf(bt['price_feature'].diff().values[1:], lags=50)
plt.show()

predX = list()
history = list(bt['price_feature'].values)
model = ARIMA(history, order=(1,1,1))
model_fit = model.fit(disp=0)
train_error = math.sqrt(sum(model_fit.resid**2)/model_fit.resid.shape[0])
for t in range(len(testX)):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predX.append(yhat)
	obs = testX[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
test_error = math.sqrt(mean_squared_error(testX, predX))
print('Train RMSE: %.3f' % train_error)
print('Test RMSE: %.3f' % test_error)

# plot
plt.plot(testX, label='Test Actual')
plt.plot(predX, label='Test Pred')
plt.title('ARIMA (p=1, d=1, q=1)')
#plt.title('LSTM')
#plt.title('GRU')
plt.ylabel('BitCoin Price')
plt.xlabel('Days')
plt.legend()
plt.show()

modelz = ['VRNN', 'LSTM', 'GRU']
for typez in modelz:    
    #data for lstm
    trainX = X[indX<'2017-09-01']
    testX = X[indX>='2017-09-01']
    trainY = Y[indY<'2017-09-03']
    testY = Y[indY>='2017-09-03']
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print ('\n\n Running Model type: {}'.format(typez))
    model = Sequential()
    if typez=='VRNN':
        model.add(SimpleRNN(units = 100, activation = 'tanh', input_shape=(1, 1), return_sequences=False))
    elif typez=='LSTM':
        model.add(LSTM(units = 100, activation = 'tanh', input_shape=(1, 1), return_sequences=False))
    elif typez=='GRU':
        model.add(GRU(units = 100, activation = 'tanh', input_shape=(1, 1), return_sequences=False))
    else:
        print('wrong option')
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=10, validation_data=(testX, testY), verbose=1)
    model.save('./savedModel')
    
    #predict
    trainHat = model.predict(trainX)
    testHat = model.predict(testX)
    
    #invert 
    trainHat = scaler.inverse_transform(trainHat)
    trainY = scaler.inverse_transform(trainY)
    testHat = scaler.inverse_transform(testHat)       
    testY = scaler.inverse_transform(testY)
    
    #rmse
    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainHat[:,0]))
    print('Train: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:,0], testHat[:,0]))
    print('Test: %.2f RMSE' % (testScore))
    
    #plot
    plt.plot(testY, label='Actual')
    #plt.plot(trainPredictPlot, label='Train Preds')
    plt.plot(testHat, label='Test Preds')
    plt.title('Model type: {}'.format(typez))
    #plt.title('LSTM')
    #plt.title('GRU')
    plt.ylabel('BitCoin Price')
    plt.xlabel('Days')
    plt.legend()
    plt.show()
    
    model.reset_states()
    #layer.reset_states()