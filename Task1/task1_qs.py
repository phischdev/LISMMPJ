from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import Ridge
import numpy

# load data
from sklearn.tree import DecisionTreeRegressor

data = numpy.genfromtxt('train.csv', delimiter=',')
data = data[1:]

xs = numpy.array(list(map(lambda li: li[2:], data)))
ys = numpy.array(list(map(lambda li: li[1], data)))

# split data
r = random.randint(0, 4294967295)  # generate a random number
X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=r)

# NOTE: does not help
'''
# PCA
pca = PCA(n_components=5).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
'''

# prediction using linear ridge regression
clf_ridge = Ridge(alpha=1.0)
clf_ridge.fit(X_train, y_train)
# predict for validation
y_pred = clf_ridge.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('linear ridge regression error: ' + repr(RMSE))

# prediction with kernel ridge regression, 3rd degree polynomial
clf_kp3 = KernelRidge(alpha=1.0, coef0=1, degree=3, kernel='poly')
clf_kp3.fit(X_train, y_train)
# predict for validation
y_pred = clf_kp3.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('Kernel (3rd degree poly) ridge regression error: ' + repr(RMSE))

# prediction with kernel ridge regression, 4th degree polynomial
clf_kp4 = KernelRidge(alpha=1.0, coef0=1, degree=4, kernel='poly')
clf_kp4.fit(X_train, y_train)
# predict for validation
y_pred = clf_kp4.predict(X_test)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
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
'''

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

# using polynomialFeature
poly2 = PolynomialFeatures(2)
X_train_p2 = poly2.fit_transform(X_train)
X_test_p2 = poly2.transform(X_test)
clf_rp2 = Ridge(alpha=1.0)
clf_rp2.fit(X_train_p2, y_train)
# predict for validation
y_pred = clf_rp2.predict(X_test_p2)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 2nd°, error: ' + repr(RMSE))

poly3 = PolynomialFeatures(3)
X_train_p3 = poly3.fit_transform(X_train)
X_test_p3 = poly3.transform(X_test)
clf_rp3 = Ridge(alpha=1.0)
clf_rp3.fit(X_train_p3, y_train)
# predict for validation
y_pred = clf_rp3.predict(X_test_p3)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 3rd°, error: ' + repr(RMSE))

# with PCA
pca_p3 = PCA(n_components=350).fit(X_train_p3)
X_train_pca = pca_p3.transform(X_train_p3)
X_test_pca = pca_p3.transform(X_test_p3)
clf_rp3_pca = Ridge(alpha=1.0)
clf_rp3_pca.fit(X_train_pca, y_train)
y_pred = clf_rp3_pca.predict(X_test_pca)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 3rd° and pca, error: ' + repr(RMSE))

clf_kpf3 = KernelRidge(alpha=1.0, coef0=1, kernel='linear')
clf_kpf3.fit(X_train_pca, y_train)
# predict for validation
y_pred = clf_kpf3.predict(X_test_pca)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('Kernel (3rd degree poly) ridge regression error: ' + repr(RMSE))
'''
gs_kpf3 = GridSearchCV(Ridge(), cv=9, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3, 1e-4, 0],
                                                  "solver": ['sag', 'lsqr']})
gs_kpf3.fit(X_train_pca, y_train)
# predict for validation
y_pred = gs_kpf3.predict(X_test_pca)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('grid search with ridge regression (poly 3rd°), and pca error: ' + repr(RMSE))
'''

'''
poly4 = PolynomialFeatures(4)
X_train_p4 = poly4.fit_transform(X_train)
X_test_p4 = poly4.transform(X_test)
clf_rp4 = Ridge(alpha=1.0)
clf_rp4.fit(X_train_p4, y_train)
# predict for validation
y_pred = clf_rp4.predict(X_test_p4)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 4th°, error: ' + repr(RMSE))
# with PCA
pca_p4 = PCA(n_components=400).fit(X_train_p4)
X_train_pca = pca_p4.transform(X_train_p4)
X_test_pca = pca_p4.transform(X_test_p4)
clf_rp4.fit(X_train_pca, y_train)
y_pred = clf_rp4.predict(X_test_pca)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 4th° and pca, error: ' + repr(RMSE))

poly5 = PolynomialFeatures(5)
X_train_p5 = poly5.fit_transform(X_train)
X_test_p5 = poly5.transform(X_test)
clf_rp5 = Ridge(alpha=1.0)
clf_rp5.fit(X_train_p5, y_train)
# predict for validation
y_pred = clf_rp5.predict(X_test_p5)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ridge regression with polynomial feature of 5th°, error: ' + repr(RMSE))
'''

# ada boost regression
rng = numpy.random.RandomState(1)
adr = AdaBoostRegressor(clf_rp3_pca, n_estimators=1000, random_state=rng)
adr.fit(X_train_pca, y_train)
y_pred = adr.predict(X_test_pca)
# error
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
print('ada boost regression: ' + repr(RMSE))

# predict for hand-in
test = numpy.genfromtxt('test.csv', delimiter=',')
test1 = test[1:]
test2 = numpy.array(list(map(lambda li: li[1:], test1)))
# NOTE: remember to change the prediction model here!!!
test_p3 = poly3.transform(test2)
test_pca = pca_p3.transform(test_p3)
# prediction = gs_kpf3.predict(test_pca)
prediction = adr.predict(test_pca)

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

# print(data)
