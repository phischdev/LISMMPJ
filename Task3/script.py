import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

df = pd.read_hdf('train.h5', 'train')
df = df._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train = df.ix[:, 'x1':].as_matrix();
y_train = df['y']
X_test = dft.ix[:, 'x1':].as_matrix();

y_train = y_train.values.reshape((-1, 1))




# define vars
input_num_units = 100
hidden1_num_units = 128
hidden2_num_units = 256
hidden3_num_units = 512
hidden4_num_units = 128
output_num_units = 45234
epochs = 20
batch_size = 128



model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(0.2),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dropout(0.2),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dropout(0.2),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dropout(0.2),
Dense(output_dim=output_num_units, input_dim=hidden4_num_units, activation='relu'),
 ])
model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size,epochs)
y_pred = model.predict_classes(X_test)
print(y_pred)
d = {'y':y_pred, 'Id': np.linspace(45324, 45324+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)
