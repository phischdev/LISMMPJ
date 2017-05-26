import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer

dfl = pd.read_hdf('train_labeled.h5', 'train')
dfl = dfl._get_numeric_data()

dfu = pd.read_hdf('train_unlabeled.h5', 'train')
dfu = dfu._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train_labeled = dfl.ix[:, 'x1':].as_matrix()
X_train_unlabeled = dfu.ix[:, 'x1':].as_matrix()
y_train_labeled = dfl['y']
# y_train_unlabeled = np.array([-1 for i in range(21000)])[np.newaxis].T
X_test = dft.ix[:, 'x1':].as_matrix()

y_train_labeled = y_train_labeled.values.reshape((-1, 1))

print('Size of X_train_labeled: ' + repr(X_train_labeled.shape))
print('Size of X_train_unlabeled: ' + repr(X_train_unlabeled.shape))
print('Size of y_train_labeled: ' + repr(y_train_labeled.shape))
# print('Size of y_train_unlabeled: ' + repr(y_train_unlabeled.shape))
print('Size of X_test: ' + repr(X_test.shape) + '\n')


'''
# merge data
X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)
y_train = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)


print('Size of X_train: ' + repr(X_train.shape))
print('Size of y_train: ' + repr(y_train.shape))
'''

nn = Classifier(
 layers=[Layer('Tanh', units=128), Layer('Sigmoid', units=128), Layer('Softmax', units=10)],
 learning_rate=0.04,
 n_iter=85,
 batch_size=10
)

nn.fit(X_train_labeled, y_train_labeled)
b = nn.predict(X_train_unlabeled)
nn.fit(X_train_unlabeled, b)
y_pred = nn.predict(X_test)

print(y_pred)

d = {'y': y_pred, 'Id': np.linspace(30000, 30000+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)

