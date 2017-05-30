import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

dfl = pd.read_hdf('train_labeled.h5', 'train')
dfl = dfl._get_numeric_data()

dfu = pd.read_hdf('train_unlabeled.h5', 'train')
dfu = dfu._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train_labeled = dfl.ix[:, 'x1':].as_matrix()
X_train_unlabeled = dfu.ix[:, 'x1':].as_matrix()
y_train = dfl['y']
X_test = dft.ix[:, 'x1':].as_matrix()

y_train = y_train.values.reshape((-1, 1))


print('Size of X_train_labeled: ' + repr(X_train_labeled.shape))
print('Size of X_train_unlabeled: ' + repr(X_train_unlabeled.shape))
print('Size of y_train: ' + repr(y_train.shape))
print('Size of X_test: ' + repr(X_test.shape) + '\n')


# derp stuff from task 3

# define vars
input_num_units = 128
hidden1_num_units = 128
hidden2_num_units = 256
hidden3_num_units = 512
hidden4_num_units = 128
output_num_units = 30000
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
model.fit(X_train_labeled,y_train,batch_size,epochs)
y_pred = model.predict_classes(X_test)
print(y_pred)
d = {'y':y_pred, 'Id': np.linspace(30000, 30000+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)

