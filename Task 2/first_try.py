import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#Load data into appropriate formats
data = numpy.genfromtxt('train.csv', delimiter=',')
data = data[1:]
data_2 = numpy.genfromtxt('test.csv',delimiter=',')
X_final = numpy.array(list(map(lambda li: li[1], data_2)))
X_train = numpy.array(list(map(lambda li: li[2:], data)))
y_train = numpy.array(list(map(lambda li: li[1], data)))


test = numpy.genfromtxt('test.csv', delimiter=',')
test1 = test[1:]
test2 = numpy.array(list(map(lambda li: li[1:], test1)))

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
    'hidden_layer_sizes':[(2,15),(4,15),(6,15),(8,15),(10,15)],
    'activation':['tanh','logistic','relu'],
    'solver':['lbfgs','sgd','adam'],
    'alpha':[0.00001, 0.0001,0.001, 0.01, 0.1, 1, 10],
    'learning_rate':['constant', 'invscaling', 'adaptive'],
    'learning_rate_init':[0.0001, 0.001, 0.01],
    'momentum':[0.7, 0.9, 0.99]
}]

#Since MLP is sensitive to feature scaling, we normalize the data
scaler = StandardScaler()

#Fit to training data
scaler.fit_transform(X_train)

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
predictions = mlp.predict(test2)










# write to file
i = 1000
prediction2 = []
for row in predictions:
    array = []
    array.append(i)
    array.append(row)
    prediction2.append(array)
    i += 1

numpy.savetxt('output.csv', prediction2, fmt="%.15f", delimiter=',', header="Id,y", comments='')