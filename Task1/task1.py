import csv
from sklearn.linear_model import Ridge
import numpy

data = []

with open('train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader)
    '''for row in spamreader:
        data.append(row)
        # print (', '.join(row))
    data = data[2:len(data)]
    for li in data:
        li = li[2:len(li)]
    data = numpy.array(data)'''
    data = numpy.genfromtxt('train.csv', delimiter = ',')
    data = data[1:len(data)]
    

    print(data)
    #data.clear();