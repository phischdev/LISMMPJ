import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score




# task 2 data reading
'''
df = pd.read_csv('train.csv', header = 0)
df = df._get_numeric_data()

dft = pd.read_csv('test.csv', header = 0)
dft = dft._get_numeric_data()

X_train = df.ix[:,'x1':].as_matrix();
y_train = df['y']
x_test = dft.ix[:,'x1':].as_matrix();

X_train = StandardScaler().fit_transform(X_train,y_train)
x_test = StandardScaler().fit_transform(x_test)
'''


# task 4 data reading
dfl = pd.read_hdf('train_labeled.h5', 'train')
dfl = dfl._get_numeric_data()

dfu = pd.read_hdf('train_unlabeled.h5', 'train')
dfu = dfu._get_numeric_data()

dft = pd.read_hdf('test.h5', 'test')
dft = dft._get_numeric_data()

X_train_labeled = dfl.ix[:, 'x1':].as_matrix()
X_train_unlabeled = dfu.ix[:, 'x1':].as_matrix()
y_train_labeled = dfl['y'].as_matrix()
# y_train_unlabeled = np.array([-1 for i in range(21000)])[np.newaxis].T
X_test = dft.ix[:, 'x1':].as_matrix()

#y_train_labeled = y_train_labeled.values.reshape((-1, 1))

print('Size of X_train_labeled: ' + repr(X_train_labeled.shape))
print('Size of X_train_unlabeled: ' + repr(X_train_unlabeled.shape))
print('Size of y_train_labeled: ' + repr(y_train_labeled.shape))
# print('Size of y_train_unlabeled: ' + repr(y_train_unlabeled.shape))
print('Size of X_test: ' + repr(X_test.shape) + '\n')



'''
h = .02  # step size in the mesh
alphas = numpy.logspace(-7, 2, 10)
names = []
for i in alphas:
    names.append('alpha ' + str(i))
classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(alpha=i, random_state=1, hidden_layer_sizes=(15,15,15), warm_start=True, solver='lbfgs'))
'''
# Define parameters to be tuned by GridSearchCV
#tuned_parameters = [{'alpha': [0.01, 0.012, 0.014, 0.016, 0.018], }]

# Git comment


# Split data into training and test part for validation
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.4)

# Find best model and fit to data
# mlp = GridSearchCV(MLPClassifier(max_iter=2100), tuned_parameters, cv=10, verbose=100)
mlp= MLPClassifier(max_iter=700, verbose=100)
mlp.fit(X_train_labeled, y_train_labeled)
fit1 = mlp.predict(X_train_unlabeled)
mlp.fit(X_train_unlabeled, fit1)


# Predict classification of test data based on model
y_pred = mlp.predict(X_test)



'''max_score = 0
max_score_index = 0
i = 0
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    if(score > max_score):
        max_score = score
        max_score_index = i
    print(name + ": " + str(score))
    i += 1
print("Best score for model " + names[max_score_index])
mlp = classifiers[max_score_index]'''




d = {'y': y_pred, 'Id': np.linspace(30000, 30000+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)

