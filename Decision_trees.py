from math import log



def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ["年龄", "有工作", "有房子", "信贷情况"]
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        # log(x[,base]) 返回以base为底的对数
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

'''
Paramenters:
    dataSet  -- 待划分的数据集
    axis -- 划分数据集的特征
    value -- 需要返回的特征的值
'''

def splitDataSet(dataSet, axis, value):
    retDateSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDateSet.append(reducedFeatVec)
    return retDateSet


def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeaature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeaature = i
    return bestFeaature



if __name__ == "__main__":
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
    print("最优特征索引值:" + str(chooseBestFeature(dataSet)))