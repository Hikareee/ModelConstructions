import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 

plt.style.use('fivethirtyeight')

#Get the stock quote 
df = web.DataReader('ANTM.JK', data_source='yahoo', start='2019-01-01', end='2021-03-26' )

#show teh data 
df

df.shape

(557, 6)

#Visualize closing price history 
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price RP', fontsize=18)
plt.show()

#Create a new dataframe with only the Close column 
data=df.filter(['Close'])
#Convert the data frame to a numpy array 
dataset = data.values 
#Number of rows to train the AI 
training_data_len = math.ceil( len(dataset) * .8)
                             
scaler = MinMaxScaler(feature_range=(0,1))
Cash = scaler.fit_transform(dataset)

#train data set 
#scale training data set 
train_data = Cash[0:training_data_len , :]
#Split data to xtrain and ytrain sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
   print(x_train)
   print(y_train)
   print()
   
#Convert the x_train and y_train 
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape

#Reshape data
x_train = np.reshape(x_train, (386, 60, 1))
x_train.shape

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))
#compile
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the bot
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create Test data set 
#new scale value array 
test_data = Cash[training_data_len - 60: , :]
#create data set 
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i,0]) 

#Convert data to a np.array 
x_test = np.array(x_test)

#Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Predicted price value 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the rmse
rmse = np.sqrt( np.mean( predictions - y_test )**2)
rmse 

#Plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Qian')
plt.xlabel('Date', fontsize=18)
plt.xlabel('Closing Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()