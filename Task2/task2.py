import numpy as np
import pandas as pd
df = pd.read_csv('train.csv', header = 0)
df = df._get_numeric_data()

dft = pd.read_csv('test.csv', header = 0)
dft = dft._get_numeric_data()

x_train = df.ix[:,'x1':]
y_train = df['y']
x_test = dft.ix[:,'x1':]




y_pred = y_train


d = {'y': y_pred, 'Id': np.linspace(900, 900+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)






