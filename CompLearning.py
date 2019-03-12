# Author:Jack Wotherspoon
# Date Created: March 1st, 2019

#import dependencies
from random import random
from csv import reader
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.kmeansClusters=[]
        self.kmeans_epoch=0

#initialize weights for nodes
    def initialize_weights(self,dataset):
        rows=len(dataset)
        input_vect = random.sample(range(0, rows), 2) #take two random points from data and call their values the weights so that algorithms are guaranteed to work
        self.weights=np.array([list(dataset[input_vect[0]]),list(dataset[input_vect[1]])]) #set the two points as the weights heading to the two output nodes

#create guess function to guess which cluster each point belongs to
    def guess(self,input_row):
        distances=[] #initializes distances to zero
        for row in range(len(self.weights)): #loops twice to test both output nodes
            weights=self.weights #get weights
            weight_row=weights[row] #grab weight for one of nodes
            distance=euclid_dist(weight_row,input_row) #calculate eucldean distance
            distances.append(distance) #add to distances array
        #return shortest distance
        winner=np.argmin(distances) #index of winning node, one with shortest distance
        return winner

#function to train network with kohenon algorithm
    def kohonen_train(self,input_row):
        distances=[]
        for row in range(len(self.weights)): #for each node
            weights=self.weights    #get weights
            weight_row = weights[row]   #grab weights for one of the nodes
            distance=euclid_dist(weight_row,input_row) #calculate euclidean distance
            distances.append(distance)  #add to distances array
        winner=np.argmin(distances)     #choose index of winning cluster
        err=error(weights[winner],input_row) #calculate error of data point
        delta=self.learnRate * (input_row - weights[winner]) #calculate weight change for winning cluster only
        weights[winner]+=delta #change weight for winning cluster only
        return err

#function for network to implement k-means network
    def k_means_train(self,input):
        weights=self.weights #get weights
        k = len(self.weights)  # number of nodes(number of clusters)
        clusters = {}       #create dictionary for clusters
        prev_cluster = {}   #create dictionary for previous clusters
        # initialize cluster dicts
        for i in range(k): #for each cluster
            clusters["cluster" + str(i)] = []
            prev_cluster["cluster" + str(i)] = []
        while True: #complete while True
            self.kmeans_epoch += 1 #count number of epochs
            for i in range(k): #for each cluster
                clusters["cluster" + str(i)] = [] #reset current points in cluster to zero
            # categorize each point by cluster
            for point in input:
                distances = [] # store distances from point to each centroid
                point = np.array(point) #convert to array type
                for i in range(k): #for each cluster
                    # calculate distance from centroids
                    distances.append(euclid_dist(weights[i],point))
                winner = np.argmin(distances) # winning index is centroid closest to point
                # if empty, initialize dimensions of array
                if clusters["cluster" + str(winner)] == []:
                    clusters["cluster" + str(winner)].append(point)
                else:
                    # using vstack to keep shape of array and coordinates of each point
                    clusters["cluster" + str(winner)] = np.vstack(
                        (clusters["cluster" + str(winner)], point))
            for i in range(k): #for each cluster
                weights[i] = np.mean(clusters["cluster" + str(i)], axis=0) # adjust centroids to mean position of all points in each cluster
            # exit if cluster for each centroid doesn't change, meaning if centroid position didn't change, terminating criteria
            if np.array_equal(clusters["cluster0"], prev_cluster["cluster0"]) and np.array_equal(clusters["cluster1"], prev_cluster["cluster1"]):
                self.kmeansClusters = copy.deepcopy(clusters) #save clusters before terminating
                break
            else:
                prev_cluster = copy.deepcopy(clusters) #save previous clusters and go again

#function to calculate euclidean distance
def euclid_dist(weight,input):
    dist=0  #initialize distance to zero
    for i in range(len(weight)): #looks at x, y, and z coords
        dist+=(float(weight[i])-float(input[i]))*(float(weight[i])-float(input[i])) #euclidean distance sum squared difference
    dist=np.sqrt(dist) #final part of euclidean distance eqn is to square root the summation
    return dist

#function to calculate squared error between two points
def error(point,centroid):
    point=np.asarray(point)    #turn point into array type
    centroid=np.asarray(centroid) #turn centroid point into array type
    err=np.square(np.subtract(point,centroid)) #calculate squared error
    return err

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:                         #for all rows in dataset set each column to a float
        row[column] = float(row[column].strip())


########BEGINNING OF TESTING#############
#loading data
filename='dataset_noclass.csv'
dataset=load_csv(filename)
dataset.pop(0)  #remove row with column titles
#convert csv strings to floats
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
#Kohonen Algorithm Implementation
learnRate=0.0001 #learning rate for network
kohonen=Neural_Network([3,2]) #create kohonen neural net
kohonen.initialize_weights(dataset) #initialize the networks weights
kohonen_init_weights = copy.deepcopy(kohonen.weights) #save weights before training
print("\nKohenon Algorithm..\n")
prev_error = 100000 #track previous epoch error
err_sum = 0 # keeps track of per epoch error
epoch = 500 # kohonen epoch counter
for i in range(epoch): # number of epochs
    if np.square(prev_error - err_sum) < .000001: # if high accuracy is achieved we can break early, terminating criteria
        print("Terminated early at epoch %d " % i)
        break
    else:
        prev_error = err_sum # update previous error
        err_sum = 0 # reset current error
        for row in dataset: #loop through each data point
            err = kohonen.kohonen_train(row) #adjust weights for each point
            err_sum += err.sum() # sum squared error
#print("Kohonen Squared Error: ", err_sum) #final error
kohonen_final_weights = copy.deepcopy(kohonen.weights) #save trained weights
print("Kohonen initial weights: ", kohonen_init_weights)
print("Kohonen Final Weights: ", kohonen_final_weights)

# sort trained points so we can plot kohonen
count=0 #used to print in txt file
clusterA=[] #used to find error in clusters
clusterB=[]
x_0 = [] # cluster 1 points
y_0 = []
z_0 = []
x_1 = [] # cluster 2 points
y_1 = []
z_1 = []
for input in dataset: #loop through each data point
    guess = kohonen.guess(input) #get the cluster point belongs to from trained network
    count+=1
    #errr=error(input,kohonen_final_weights)
    #errr_sum=0
    #errr_sum += errr.sum()
    #print(errr_sum)
    if guess == 0: #if its first cluster add coords
        x_0.append(input[0])
        y_0.append(input[1])
        z_0.append(input[2])
        clusterA.append(input)
    else:          #if its second cluster for kohonen add coords
        x_1.append(input[0])
        y_1.append(input[1])
        z_1.append(input[2])
        clusterB.append(input)

#get errors per cluster and total
errorA=0
for cluster in clusterA:
    errA=error(cluster,kohonen_final_weights[0])
    errorA+=errA.sum()
print("Sum Squared Error Kohonen Cluster 0:",errorA)
errorB=0
for cluster in clusterB:
    errB=error(cluster,kohonen_final_weights[1])
    errorB+=errB.sum()
print("Sum Squared Error Kohonen Cluster 1:",errorB)
kohonen_error=errorA+errorB
print("Total Kohonen Error: ",kohonen_error)

#create fig
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #initialize 3d

ax.scatter(x_0, y_0, z_0, c='r') #scatter first cluster of kohonen onto fig
ax.scatter(x_1, y_1, z_1, c='b') #scatter second cluster of kohonen onto fig

#create labels for kohonen diagram
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("Kohonen Results")


#K-means Algorithm Implementation
k_means=Neural_Network([3,2]) #initializing k-means ANN
k_means.initialize_weights(dataset) # initialize weights for k-means
k_means_init_weights = copy.deepcopy(k_means.weights) # save initial weights for k-means
k_means.k_means_train(dataset) #train k-means
k_means_final_weights = copy.deepcopy(k_means.weights) #copy final weights
print("\nK-Means Algorithm...")
print("\nK-Means Initial Weights: ", k_means_init_weights)
print("\nK-Means Final Weights: ", k_means_final_weights)
print("\nK_means epoch: ", k_means.kmeans_epoch)
# create data points for plotting k-means
count=0 #used to print in txt file
clusterC=[] #used to find errors in clusters
clusterD=[]
x_2 = [] # cluster 1 points
y_2 = []
z_2 = []
x_3 = [] # cluster 2 points
y_3 = []
z_3 = []

for input in dataset:  #loop through to place each point from data
    guess = k_means.guess(input)
    count+=1
    if guess == 0:  #classify the points for cluster 1 of k-means
        x_2.append(input[0])
        y_2.append(input[1])
        z_2.append(input[2])
        clusterC.append(input)
    else:           #classify the points for cluster 2 of k-means
        x_3.append(input[0])
        y_3.append(input[1])
        z_3.append(input[2])
        clusterD.append(input)

#get errors per cluster and total
errorC=0
for cluster in clusterC:
    errC=error(cluster,k_means_final_weights[0])
    errorC+=errC.sum()
print("Sum Squared Error K-means Cluster 0:",errorC)
errorD=0
for cluster in clusterD:
    errD=error(cluster,k_means_final_weights[1])
    errorD+=errD.sum()
print("Sum Squared Error K-means Cluster 1:",errorD)
kmeans_error=errorC+errorD
print("Total K-means Error: ",kmeans_error)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x_2, y_2, z_2, c='r')   #cluster 1 of k-means is red
ax1.scatter(x_3, y_3, z_3, c='b')   #cluster 2 of k-means in blue

#label diagram
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title("K-means Results")

plt.show() #print plots
