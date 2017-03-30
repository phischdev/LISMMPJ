from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import Ridge
import numpy

# load data
data = numpy.genfromtxt('train.csv', delimiter=',')
data = data[1:]

xs = numpy.array(list(map(lambda li: li[2:], data)))
ys = numpy.array(list(map(lambda li: li[1], data)))

# split data
r = random.randint(0, 4294967295) # generate a random number
X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=r)

# prediction using linear ridge regression
clf_ridge = Ridge(alpha=1.0)
clf_ridge.fit(X_train, y_train)
# predict for validation
y_pred = clf_ridge.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('linear ridge regression error: ' + repr(RMSE))


# prediction with kernel ridge regression, 3rd degree polynomial
clf_kp3 = KernelRidge(alpha=1.0, coef0=1, degree=3, kernel='poly')
clf_kp3.fit(X_train, y_train)
# predict for validation
y_pred = clf_kp3.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('Kernel (3rd degree poly) ridge regression error: ' + repr(RMSE))


# prediction with kernel ridge regression, 4th degree polynomial
clf_kp4 = KernelRidge(alpha=1.0, coef0=1, degree=4, kernel='poly')
clf_kp4.fit(X_train, y_train)
# predict for validation
y_pred = clf_kp4.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('Kernel (4th degree poly) ridge regression error: ' + repr(RMSE))


# NOTE: doesn't really help any more than 4th degree polynomial
'''
# prediction with kernel ridge regression, 5th degree polynomial
clf_kp5 = KernelRidge(alpha=1.0, coef0=1, degree=4, kernel='poly')
clf_kp5.fit(X_train, y_train)
# predict for validation
y_pred = clf_kp5.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('Kernel (5th degree poly) ridge regression error: ' + repr(RMSE))
'''


# NOTE: is a bit worse off than grid search with kernel ridge regression, poly 3rd° most of the time
# takes a long time to compute, so I'm commenting thing part out
'''
# grid search with kernel ridge regression, rbf
gs_rbf = GridSearchCV(KernelRidge(kernel='rbf', gamma=1.0), cv=9, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                                                          "gamma": numpy.logspace(-2, 2, 5)})
gs_rbf.fit(X_train, y_train)
# predict for validation
y_pred = gs_rbf.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('grid search with kernel ridge regression (rbf) error: ' + repr(RMSE))
'''


# grid search with kernel ridge regression, poly 3rd°
gs_kp3 = GridSearchCV(KernelRidge(kernel='poly', degree=3), cv=9, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                                                          "gamma": numpy.logspace(-2, 2, 5)})
gs_kp3.fit(X_train, y_train)
# predict for validation
y_pred = gs_kp3.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('grid search with kernel ridge regression (poly 3rd°) error: ' + repr(RMSE))


# NOTE: is a bit worse off than grid search with kernel ridge regression, poly 3rd° most of the time
# takes a long time to compute, so I'm commenting thing part out
'''
# grid search with kernel ridge regression, poly 4th°
gs_kp4 = GridSearchCV(KernelRidge(kernel='poly', degree=4), cv=9, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                                                          "gamma": numpy.logspace(-2, 2, 5)})
gs_kp4.fit(X_train, y_train)
# predict for validation
y_pred = gs_kp4.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred)**0.5
print('grid search with kernel ridge regression (poly 4th°) error: ' + repr(RMSE))
'''


# predict for hand-in
test = numpy.genfromtxt('test.csv', delimiter=',')
test1 = test[1:]
test2 = numpy.array(list(map(lambda li: li[1:], test1)))

prediction = gs_kp3.predict(test2)

# write to file
i = 900
prediction2 = []
for row in prediction:
    array = []
    array.append(i)
    array.append(row)
    prediction2.append(array)
    i += 1

numpy.savetxt('output.csv', prediction2, fmt="%.15f", delimiter=',', header="Id,y", comments='')

#print(data)
