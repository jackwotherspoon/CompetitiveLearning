# Author:Jack Wotherspoon
# Date Created: March 1st, 2019

from random import random
from csv import reader
import numpy as np
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

def euclid_dist(weight_vec,input_vec):
    dist = np.sqrt(np.sum(np.square(np.subtract(input_vec, weight_vec))))
    return dist

def kohonen_guess(self, input_arr):
    node_dists = []
    for key in self.layer_weights.keys():
        weights_mat = self.layer_weights[key]
        num_rows = weights_mat.shape[0]
        for row in range(num_rows):
            weight_row = weights_mat[row,]
            dist = euclid_dist(weight_row, input_arr)
            node_dists.append(dist)
    winner = np.argmin(node_dists)  # returns the index of the smallest distance
    return winner