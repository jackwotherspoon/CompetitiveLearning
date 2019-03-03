# Author:Jack Wotherspoon
# Date Created: March 1st, 2019

from random import random
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file: #if file can be read
        csv_reader = reader(file)
        for row in csv_reader: #loop through each row of file and add it
            if not row:
                continue
            dataset.append(row)
    return dataset

filename='dataset_noclass.csv'
dataset=load_csv(filename)
dataset.pop(0)  #remove row with column titles
print(dataset)
