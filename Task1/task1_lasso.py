from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import Ridge
from sklearn import linear_model
import numpy
# load data
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV

data = numpy.genfromtxt('train.csv', delimiter=',')
data = data[1:]
data_2 = numpy.genfromtxt('test.csv',delimiter=',')
X_final = numpy.array(list(map(lambda li: li[1], data_2)))
xs = numpy.array(list(map(lambda li: li[2:], data)))
ys = numpy.array(list(map(lambda li: li[1], data)))


test = numpy.genfromtxt('test.csv', delimiter=',')
test1 = test[1:]
test2 = numpy.array(list(map(lambda li: li[1:], test1)))




polys = PolynomialFeatures(3)
XS = polys.fit_transform(xs)
T2 = polys.fit_transform(test2)
model = LassoCV(cv=20)

model.fit(XS,ys)
prediction = model.predict(T2)



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