from numpy import *
import operator
from os import listdir

def img2vector(filename) :
   returnVect = zeros((1,1024))
   fr = open(filename)
   frlines = fr.readlines()
   m = len(frlines)
   i = 0
   for string in frlines :
   	  for j in range(len(string)-1) :
           returnVect[0][i*(len(string)-1)+j] = int(string[j])
	  i+=1
   return returnVect

def KNN(inX, dataSet, labels, k) : # inX : input, labels : datasets' label
    m = dataSet.shape[0]
    diffMat = dataSet - tile(inX, (m, 1))
    diffMat = diffMat ** 2
    distList = diffMat.sum(axis=1)
    orderList = distList.argsort()

    labelCount = {}
    bestlabel = labels[0];
    bestlabelcount = -1;
    for i in range(k) :
       labelCount[labels[orderList[i]]] = labelCount.get(labels[orderList[i]], 0)+1
       if (bestlabelcount < labelCount[labels[orderList[i]]]) :
           bestlabelcount = labelCount[labels[orderList[i]]]
           bestlabel = labels[orderList[i]]
    return bestlabel                  

def normalize(dataSet) :
   m = dataSet.shape[0]
   mins = dataSet.min(0)
   maxs = dataSet.max(0)
   ranges = maxs-mins
   returnSet = (dataSet - tile(mins, (m,1)))/tile(ranges, (m,1))
   return returnSet

def main() :

	## Training Session ##
	labels = []
	trainingAddress = "D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\trainingDigits" # to user's Address
	fileList = listdir(trainingAddress)
	m = len(fileList)
	dataSet = zeros((m, 1024))

	for i in range(m) :
		fileName = fileList[i]
		dataSet[i, :] = img2vector(trainingAddress+"\\"+fileName)
		labels.append(int(fileName.split('_')[0]))

	## Test Session ##
	errorCount = 0.0;
	testAddress = "D:\Downloads\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\testDigits" # to user's Address
	testFileList = listdir(testAddress)
	m = len(testFileList)

	for i in range(m) :
		fileName = testFileList[i]
		KNNResult = KNN(img2vector(testAddress+"\\"+fileName), dataSet, labels, 3)
		answerResult = int(fileName.split('_')[0])
		print "KNN result : %d, Real answer : %d \n" % (KNNResult, answerResult)
		if KNNResult!=answerResult :
			errorCount+=1.0

	print "Total error # : %d\n" % int(errorCount)
	print "Total error rate : %f\n" % (float(errorCount)/m)

main()