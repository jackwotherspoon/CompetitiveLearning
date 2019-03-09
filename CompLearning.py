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

def k_means_learn(self, all_inputs):
        # kinda broke the expandability of my code by using this index
        # take the only weight layer
        weights_mat = self.layer_weights["weights1"]
        k = weights_mat.shape[0]  # number of nodes aka number of clusters
        clusters = {}
        prev_cluster = {}
        # initialize cluster lists
        for i in range(k):
            clusters["cluster" + str(i)] = []
            prev_cluster["cluster" + str(i)] = []
        while True:
            self.k_means_epoch += 1
            for i in range(k):
                clusters["cluster" + str(i)] = []
            # Categorize each point by cluster
            for point in all_inputs:
                node_dists = []
                point = np.array(point)
                for i in range(k):
                    # calculate distance from centroids
                    node_dists.append(euclid_dist(weights_mat[i,], point))
                winner = np.argmin(node_dists)
                # if empty, initialize dimensions of ndarray
                if clusters["cluster" + str(winner)] == []:
                    clusters["cluster" + str(winner)].append(point)
                else:
                    # creates ndarray of cluster points
                    clusters["cluster" + str(winner)] = np.vstack(
                        (clusters["cluster" + str(winner)], point))
            for i in range(k):
                weights_mat[i,] = np.mean(clusters["cluster" + str(i)])

            # ( == prev_cluster["cluster0"]) and (clusters["cluster1"] == prev_cluster["cluster1"]):
            if np.array_equal(clusters["cluster0"], prev_cluster["cluster0"]) and np.array_equal(clusters["cluster1"],
                                                                                                 prev_cluster[
                                                                                                     "cluster1"]):
                self.k_means_clusters = copy.deepcopy(clusters)
                break
            else:
                prev_cluster = copy.deepcopy(clusters)

