from numpy import *
from math import exp

## This Code implemented Logistic Regression with either GD and SGD ##
## First Part : find best seperating line ##
## Second Part : Use Horse Colic Data to Test Logistic Regrssion ##

def sig(x) :
	if x>500 : return 1
	elif x<-500 : return 0
	else : return 1.0/(1.0+exp(-x))

def generateDataSet() :
	dataSet = []; ySet = []
	fr = open("D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch05\\testSet.txt")
	for line in fr.readlines() :
		lineArr = line.strip().split()
		dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
		ySet.append(int(lineArr[2]))
	return dataSet, ySet

def gradientDescent(dataSet, yList) : # assume that dataSet is numpy array + x0 terms added 
	m = shape(dataSet)[0]
	n = shape(dataSet)[1]

	# weight vectors start with ones 
	weightVec = ones((n,1))
	iterationNum = 2000
	alpha = 0.001

	for i in range(iterationNum) :
		newWeightVec = weightVec
		for j in range(m) :
			newWeightVec = newWeightVec + alpha * dataSet[j].T * (yList[j]-sig(float(dot(dataSet[j], weightVec))))
		weightVec = newWeightVec

	return weightVec

def stochasticGradientDescent(dataSet, yList) : # alpha changes during cycle
	m = shape(dataSet)[0]
	n = shape(dataSet)[1]

	# weight vectors start with ones 
	iterationNum = 500
	weightVec = ones((n,1))
	alpha = 0.001

	for j in range(iterationNum) :
		for i in range(m) :
			alpha = 4 / (1.0+j+i) + 0.01
			randIndex = int(random.uniform(0, m)) # data chosen randomly
			weightVec = weightVec + alpha * dataSet[randIndex].T * (yList[randIndex] - sig(dot(dataSet[randIndex], weightVec)))

	return weightVec

def plotResult(weightVec) :
	import matplotlib.pyplot as plt
	dataSet, ySet = generateDataSet()
	yesSetX = []; yesSetY = []; noSetX = []; noSetY = [];
	for i in range(len(ySet)) :
		if ySet[i] == 1 : 
			yesSetX.append(dataSet[i][1]); yesSetY.append(dataSet[i][2])
		else :
			noSetX.append(dataSet[i][1]); noSetY.append(dataSet[i][2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(yesSetX, yesSetY, s=30, c='red', marker='s')
	ax.scatter(noSetX, noSetY, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = ((-weightVec[0] - weightVec[1]*x) / weightVec[2]).tolist()[0]
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.show()

def generateHorseColicData() :
	## This Horse Colic Data is Preprocessed ##
	## omitted values are replaced with value 0 ##
	frTrain = open("D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch05\\horseColicTraining.txt")
	frTest = open("D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch05\\horseColicTest.txt")
	trainDataSet = []; trainY = []; testDataSet = []; testY = [];

	for line in frTrain.readlines() :
		lineArr = line.strip().split()
		inputList = [1.0]
		for i in range(len(lineArr)-1) :
			inputList.append(float(lineArr[i]))
		trainDataSet.append(inputList)
		trainY.append(float(lineArr[-1]))

	for line in frTest.readlines() :
		lineArr = line.strip().split()
		inputList = [1.0]
		for i in range(len(lineArr)-1) :
			inputList.append(float(lineArr[i]))
		testDataSet.append(inputList)
		testY.append(float(lineArr[-1]))

	return trainDataSet, trainY, testDataSet, testY

def horseColicTest() :
	
	trainDataSet, trainY, testDataSet, testY = generateHorseColicData()
	weightVec = stochasticGradientDescent(mat(trainDataSet), trainY)
	errorCount = 0.0

	for i in range(len(testDataSet)) :
		data = mat(testDataSet[i])
		sigVal = sig(dot(data, weightVec))
		testResult = 0.0
		if sigVal > 0.5 : testResult = 1.0
		realResult = int(testY[i])
		print "Test Result : %d, Real Result : %d" % (testResult, realResult)
		if testResult!=realResult : errorCount+=1.0

	print "Total Error Num : %d" % int(errorCount)
	print "Total Error Rate : %f" % (float(errorCount)/float(len(testDataSet)))

## 2-D Data Plot ##
# dataSet, ySet = generateDataSet()
# weightVec = gradientDescent(mat(dataSet), ySet)
# weightVec = stochasticGradientDescent(mat(dataSet), ySet)
# plotResult(weightVec)

## Real Data Categorizing with Logistic Regression - Horse Colic Data ##
horseColicTest()