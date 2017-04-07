import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('train.csv', header = 0)
df = df._get_numeric_data()

dft = pd.read_csv('test.csv', header = 0)
dft = dft._get_numeric_data()

y_train = df['y']


x_total = df.ix[:,'x1':] + dft.ix[:,'x1':]
x_total_scaled = StandardScaler.fit_transform(x_total)
x_train = x_total_scaled[0:999]
x_test = x_total_scaled[1000:3999]



'''h = .02  # step size in the mesh
alphas = numpy.logspace(-7, 2, 10)
names = []
for i in alphas:
    names.append('alpha ' + str(i))
classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(alpha=i, random_state=1, hidden_layer_sizes=(15,15,15), warm_start=True, solver='lbfgs'))
'''
#Define parameters to be tuned by GridSearchCV
tuned_parameters = [{
    'solver':['sgd','adam','lbfgs'],
    'alpha':[0.00001, 0.0001,0.001, 0.01, 0.1, 1, 10],
}]

#Git comment

#Since MLP is sensitive to feature scaling, we normalize the data
scaler = StandardScaler()

#Fit to training data
scaler.fit_transform(X_train)
scaler.fit_transform(y_train)
scaler.fit_transform(x_test)

#Split data into training and test part for validation
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.4)

#Find best model and fit to data
mlp = GridSearchCV(MLPClassifier(), tuned_parameters, cv=20)
mlp.fit(X_train, y_train)



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


#Predict classification of test data based on model
y_pred = mlp.predict(x_test)



d = {'y': y_pred, 'Id': np.linspace(1000, 1000+y_pred.size-1, num=y_pred.size)}
dfp = df = pd.DataFrame(data=d)
dfp.Id = df.Id.astype(int)
dfp.to_csv('output.csv', index=False)

