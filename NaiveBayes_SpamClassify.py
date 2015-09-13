from numpy import *
from math import log
from os import listdir
import re
	
## This code uses Laplace-Smoothed Naive Bayes to perform Spam Classification ##
## Multi-Labels OK ##
## choose max_y P(y|x) => max_y P(x|y)P(y) = max_y (logP(y) + sigma logP(x_i|y)) ##

def makeDictionary(dataSet) :
	myDictionary = []
	for text in dataSet :
		for word in text :
			myDictionary.append(word)
	return list(set(myDictionary))

def txt2text(filename) :
	regEx = re.compile('\\W*')
	filetext = open(filename).read()
	returntext = [tok.lower() for tok in regEx.split(filetext) if len(tok)>0]
	return returntext

def text2wordvec(dictionary, text) : # text : list of words
	returnVec = [0]*len(dictionary);
	for word in text :
		if word in dictionary : returnVec[dictionary.index(word)]+=1
	return returnVec

def trainNB(dataSet, labels) : # label = 0, 1, 2, .. , n-1
 	m = shape(dataSet)[0]
	labelNum = len(set(labels))
	dictionary = makeDictionary(dataSet)
	## Laplace Smoothing : initialize with 1 and labelNum
	wordCountArray = tile([1], (labelNum, len(dictionary)))# countArray[i][j] = # of j if label is i
	labelCountArray = tile([float(labelNum)], (labelNum, 1))

	for i in range(m) :
		wordvec = text2wordvec(dictionary, dataSet[i])
		wordCountArray[labels[i]] += wordvec
		labelCountArray[labels[i]][0] += 1.0

	wordProbArray = wordCountArray / tile(labelCountArray, (1, len(dictionary)))
	labelProbArray = ( (labelCountArray - tile([float(labelNum-1)], (labelNum, 1)))/ tile([m+labelNum], (labelNum, 1)) ).T

	return wordProbArray, labelProbArray

def generateData() :
	trainData = [];
	trainLabel = [];
	testData = [];
	testLabel = [];
	## cross-validation : 7:3
	for i in range(26) :
		if i==0 : continue
		text = txt2text("D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch04\email\\ham\\%d.txt" % i) # to user's Address
		trainData.append(text)
		trainLabel.append(0)
		text = txt2text("D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch04\email\\spam\\%d.txt" % i) # to user's Address
		trainData.append(text)
		trainLabel.append(1) # 0 = ham, 1 = spam
	for i in range(15) :
		randIndex = int(random.uniform(0, len(trainData)))
		testData.append(trainData[randIndex])
		testLabel.append(trainLabel[randIndex])
		del(trainData[randIndex])
		del(trainLabel[randIndex])

	return trainData, trainLabel, testData, testLabel

def getNBLabel(testWordVec, labelNum, wordProbArray, labelProbArray) :
	bestAns = -99999.0
	bestLabel = -1
	for i in range(labelNum) :
		nowAns = log(labelProbArray[0][i])
		for j in range(len(testWordVec)) :
			if testWordVec[j]>0 : nowAns += log(wordProbArray[i][j])
		if (nowAns>bestAns) :
			bestAns = nowAns
			bestLabel = i
	return bestLabel

def testNB(trainData, trainLabel, testData, testLabel) :
	dictionary = makeDictionary(trainData)
	wordProbArray, labelProbArray = trainNB(trainData, trainLabel)
	labelNum = len(set(trainLabel))
	testNum = len(testData)
	errorCount = 0.0
	for i in range(testNum) :
		NBlabel = getNBLabel(text2wordvec(dictionary, testData[i]), labelNum, wordProbArray, labelProbArray)
		print "NB Label : %d, Real Label : %d" % (NBlabel, testLabel[i])
		if NBlabel != testLabel[i] : 
			errorCount += 1.0
			print testData[i]
	print "Error Rate : %f" % (errorCount/testNum)

trainData, trainLabel, testData, testLabel = generateData()
testNB(trainData, trainLabel, testData, testLabel)