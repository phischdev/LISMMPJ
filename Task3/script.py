import pandas as pd
import numpy as np


df = pd.read_hdf('train.csv', 'train')
df = df._get_numeric_data()

dft = pd.read_hdf('test.csv', 'test')
dft = dft._get_numeric_data()

X_train = df.ix[:,'x1':].as_matrix();
y_train = df['y']
x_test = dft.ix[:,'x1':].as_matrix();