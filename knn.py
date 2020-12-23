
import pandas as pd
import argparse
import numpy as np

#L2 gets the euclidean distance of x and y
def L2(x, y): 
	if len(x) != len(y): 
		print("length of x and y are not the same")
		exit(1) 
	sum = 0
	for i in range(len(x)):
		sum += (x[i] - y[i]) ** 2 	
	return sum ** .5

	 
def L1(x, y): 
	if len(x) != len(y): 
		print("length of x and y are not the same")
		exit(1) 
	sum = 0
	for i in range(len(x)):
		sum += abs(x[i] - y[i])  	
	return sum

def Linf(x,y): 
	xx = np.array(x) 
	yy = np.array(y) 

	return max(abs(xx - yy))

def getMethod(): 
	# returns K, method of measuring distance 
	# default values: 3, L2
	K = 3
	distance = L2

	parser = argparse.ArgumentParser(description = 'k nearest neighbor')
	parser.add_argument('-k', "--K", type = int) 
	parser.add_argument('-m', "--method", type = str) 
	args = parser.parse_args()

	if(type(args.K) == int):
		K = args.K

	if args.method == "L1": 
		distance = L1
	elif args.method == "L2":
		distance = L2
	elif args.method == "Linf": 
		distance = Linf

	return K, distance

def read_csv(filePath): 
	# this reads the training data and returns a array of training data
	# of the form [classification, arr(features)] 
	data = pd.read_csv(filePath)
	df = pd.DataFrame(data)

	dataPoints = np.array(df)
	return list(map(list, zip(dataPoints[:,0], dataPoints[:,1:])))


def knn(D,K,x, distance):
	# this returns a classification based on K nearest neighbors with specified distance method
	d = [x[:] for x in D] #this creates a deep copy of D
	for data in d: 
		data.append(distance(data[1], x[1]))  #this appends the distance of data 
	d.sort(key = lambda x : x[2]) #sort by distance which was just appended
	sum = 0
	for i in range(K): 
		sum += d[i][0] 
	if sum < 0: 
		return -1
	elif sum > 0:
		return 1

if __name__ == "__main__":
	trainData = read_csv("knn_train.csv")
	testData = read_csv("knn_test.csv")
	K, method = getMethod()
	for x in testData: 
		classification = knn(trainData, K, x, method)
		print(classification)

