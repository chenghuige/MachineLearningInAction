
## This code generates an ID3 Decision Tree and plot this tree by matplotlib ##
## TreePlotting Codes are from text, rest by YuKiSa ##

from numpy import *
import matplotlib.pyplot as plt
from math import log

## Tree Plot Section ##
## tree plot codes are from MLinA text ##
## @author : Peter Harington ##

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


## ID3 Decision Tree Section ##
## dataSet = [[feature1, feature2, ..., featureN, label], ... [feature1, ..., label]]

def calcShannonEnt(dataSet) :
	m = len(dataSet)
	labelCount = {};
	for featureVec in dataSet :
		labelCount[featureVec[-1]] = labelCount.get(featureVec[-1], 0.0)+1.0;
	totalEnt = 0.0;
	for label in labelCount.keys() :
		p = labelCount[label]/float(m)
		totalEnt -= p*log(p, 2)
	return totalEnt

def dataSplit(dataSet, divFeatureIndex, value) :
	returnList = []
	for featureVec in dataSet :
		if featureVec[divFeatureIndex]==value : 
			newfeatVec = featureVec[:divFeatureIndex]
			newfeatVec.extend(featureVec[divFeatureIndex+1:])
			returnList.append(newfeatVec)
	return returnList

def chooseBestFeature(dataSet) :
	bestEntropy = calcShannonEnt(dataSet)
	bestFeatureIndex = -1
	for i in range(len(dataSet[0])-1) :
		featList = [example[i] for example in dataSet] # fetch all available featureList
		featList = set(featList)
		nowEntropy = 0.0
		for feature in featList :
			nowEntropy += calcShannonEnt(dataSplit(dataSet, i, feature))
		if (bestEntropy > nowEntropy) :
			bestEntropy = nowEntropy
			bestFeatureIndex = i
	return bestFeatureIndex

def majorityVote(dataSet) :
	labelCount = {}
	bestCount = -1
	bestLabel = dataSet[0][-1]
	for vec in dataSet :
		label = vec[-1]
		labelCount[label] = labelCount.get(label, 0)+1
		if bestCount < labelCount[label] :
			bestCount = labelCount[label]
			bestLabel = label
	return bestLabel

def generateTree(dataSet, labels) :
	
	## stop condition ##
	# condition 1 : no more benefit on dividing
	bestFeatureIndex = chooseBestFeature(dataSet)
	if bestFeatureIndex==-1 : return majorityVote(dataSet)

	# condition 2 : only one kind left
	kindList = [vec[-1] for vec in dataSet]
	kindSet = set(kindList)
	if (len(kindSet)==1) : return kindSet[0]

	# condition 3 : no more labels
	if (len(labels)==0) : return majorityVote(dataSet)
	
	returnTree = {}
	returnTreeElement = {}
	availableFeatureValSet = [vec[bestFeatureIndex] for vec in dataSet]
	availableFeatureValSet = set(availableFeatureValSet)
	newLabels = labels[:bestFeatureIndex]
	newLabels.extend(labels[bestFeatureIndex+1:])
	for val in availableFeatureValSet :
		returnTreeElement[val] = generateTree(dataSplit(dataSet, bestFeatureIndex, val), newLabels)	

	returnTree[labels[bestFeatureIndex]] = returnTreeElement

	return returnTree


def myData() :
	return [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]

print createPlot( generateTree(myData(), ['feature1', 'feature2']))