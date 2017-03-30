import csv
from sklearn.linear_model import Rich
import numpy

with open('train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        print (', '.join(row))