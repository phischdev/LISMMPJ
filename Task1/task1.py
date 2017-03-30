import csv
from sklearn.linear_model import Ridge
import numpy

data = []

with open('train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader)
    for row in spamreader:
        head, *tail = row;
        data.append((head, tail));
        # print (', '.join(row))

data.clear();