import pandas as pd
import numpy as np
import tables as tb
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

df = pd.read_hdf('train.h5', 'train')
df = df._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train = df.ix[:,'x1':].as_matrix();
y_train = df['y']
x_test = dft.ix[:,'x1':].as_matrix();

df = pd.read_hdf('train.h5', 'train')
df = df._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train = df.ix[:,'x1':].as_matrix();
y_train = df['y']
x_test = dft.ix[:,'x1':].as_matrix();

print('Size of X_train: ' + repr(X_train.shape))
print('Size of y_train: ' + repr(y_train.shape))
print('Size of x_test: ' + repr(x_test.shape) + '\n')


# rearrange data to look like an image
X_train = np.reshape(X_train, (len(X_train), 10, 10))
x_test = np.reshape(x_test, (len(x_test), 10, 10))

print('Size of X_train: ' + repr(X_train.shape))
print('Size of x_test: ' + repr(x_test.shape) + '\n')


X_train = X_train.reshape(X_train.shape[0], 1, 10, 10)
x_test = x_test.reshape(x_test.shape[0], 1, 10, 10)

print('Size of X_train: ' + repr(X_train.shape))
print('Size of x_test: ' + repr(x_test.shape) + '\n')

# Convert data type and normalize values
X_train = X_train.astype('float32')
x_test = x_test.astype('float32')
max_value1 = np.max(X_train)
max_value2 = np.max(x_test)
max_value = np.maximum(max_value1, max_value2)
X_train /= max_value
x_test /= max_value


# Convert 1-dimensional class arrays to 5-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 5)


# Declare sequential model
model = Sequential()

# CNN input layer
model.add(Convolution2D(128, 3, 3, activation='relu', input_shape=(1, 10, 10), dim_ordering='th'))

# adding more layers
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax')) # output layer size is 5 like the number of our classes


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit Keras model
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

y_pred = model.predict(x_test, batch_size=32, verbose=1)
y_pred = np.argmax(y_pred, 1)

d = {'y': y_pred, 'Id': np.linspace(45324, 45324+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)


print('Complete!')