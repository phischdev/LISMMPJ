from sklearn.linear_model import Ridge
import numpy

data = numpy.genfromtxt('train.csv', delimiter = ',')
data = data[1:]

xs = numpy.array(list(map(lambda li : li[2:], data)))
ys = numpy.array(list(map(lambda li: li[1], data)))

clf = Ridge(alpha=1.0)
clf.fit(xs, ys)


test = numpy.genfromtxt('test.csv', delimiter= ',')
test1 = test[1:]
test2 = numpy.array(list(map(lambda li : li[1:], test1)))

prediciton = clf.predict(test2)



i = 900
prediciton2 = []
for row in prediciton:
        array = []
        array.append(i)
        array.append(row)
        prediciton2.append(array)
        i += 1

numpy.savetxt('output.csv', prediciton2, fmt="%.15f", delimiter=',')





print(data)


    #data.clear();