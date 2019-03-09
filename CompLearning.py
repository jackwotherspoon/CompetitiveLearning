# Author:Jack Wotherspoon
# Date Created: March 1st, 2019

from random import random
from csv import reader
import numpy as np
import random
import matplotlib.pyplot as plt
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

#create a neural network class so we can use it on Kohonen and K-means algorithm
class Neural_Network():
    def __init__(self,arr):
        self.layers=len(arr)
        self.learnRate=learnRate
        self.kohonenDist=[]
        self.kMeansDist=[]

    def initialize_weights(self,dataset):
        rows=len(dataset)
        input_vect = random.sample(range(0, rows), 2)
        self.weights=(dataset[input_vect[0]],dataset[input_vect[1]])

    def kohonen_layer(self,row):
        distances=[]
        for input_row in range(len(self.weights)):
            weight_row=self.weights[input_row]
            distance=euclid_dist(weight_row,row)
            distances.append(distance)
        #return shortest distance
        winner=min(distances)
        return winner

def euclid_dist(weight,input):
    dist=0
    for i in range(len(weight)):
        dist+=(float(weight[i])-float(input[i]))*(float(weight[i])-float(input[i]))
    dist=np.sqrt(dist)
    return dist

filename='dataset_noclass.csv'
dataset=load_csv(filename)
dataset.pop(0)  #remove row with column titles
learnRate=0.1
net=Neural_Network([3,2])
net.initialize_weights(dataset)
for input in dataset:
    guess=net.kohonen_layer(input)
    print(guess)
print(net.weights)