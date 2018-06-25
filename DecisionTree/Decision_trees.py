from math import log
import operator
import pickle
import matplotlib.pyplot as plt


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
    axis -- 划分数据集的特征 {[0, 1, 2, 3]:["年龄", "有工作", "有房子", "信贷情况"]}
    value -- 需要返回的特征的值
'''


def splitDataSet(dataSet, axis, value):
    retDateSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDateSet.append(reducedFeatVec)
    return retDateSet


## 选择最优特征
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0  # 信息增熵
    bestFeature = -1  # 最优特征索引值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set{}集合,元素不可重复
        newEntropy = 0  # 经验条件熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            # 经验条件熵 = 条件概率 * 经验熵
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益:经验熵 - 经验条件熵
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


## 统计classList中出现最多的元素
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # sort:在本地进行排序,不返回副本  sorted:返回副本,原始输入不变
    # operator.itemgetter(n) 获取第n个位置的元素
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount


# 创建决策树
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同,则停止分类
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataSet)  # 选取最优特征
    print("最优特征索引值:" + str(bestFeat))
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优标签生成树
    del (labels[bestFeat])  # 删除已经使用的标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中的所有最优属性值
    uniqueVals = set(featValues)  # 去除所有重复的属性值
    for value in uniqueVals:  # 遍历特征创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


# 获取决策树叶子节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    # iter()创建一个迭代器对象
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制节点
'''
    nodeTxt - 节点名
    centerPt - 文本中心点
    parentPt - 标注箭头的位置(指向文本的点)
    nodeType - 节点格式
'''


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle='<-')
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xytext=centerPt, va='center',ha='center',
                            bbox=nodeType, arrowprops=arrow_args)


# 标注有向边属性值(父节点和子节点的中间信息)
# 矢量边的文本标注
'''
    cntrPt, parentPt - 用于计算标注位置
    txtString - 标注内容
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (cntrPt[0] + parentPt[0])/2
    yMid = (cntrPt[1] + parentPt[1])/2
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)



# 绘制决策树
'''
    myTree - 决策树
    parentPt - 标注的内容
    nodeTxt - 节点名
'''


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    # 定义节点属性,round4表示矩形方框
    leafNode = dict(boxstyle='round4', fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1 + float(numLeafs))/2/plotTree.totalW, plotTree.yOff )
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff+ 1/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD




# 创建绘制面板
'''
    inTree - 决策树(字典)
'''


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))    # 决策树的广度
    plotTree.totalD = float(getTreeDepth(inTree))   # 决策树的深度
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1
    plotTree(inTree, (0.5, 1), '')
    plt.show()


'''
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征
    testVec - 测试数据列表, 顺序对应最优特征标签
'''
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = 0
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(myTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(myTree, fw)

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    # print(splitDataSet(dataSet, 2, 1))
    # print(calcShannonEnt(dataSet))
    # print("最优特征索引值:" + str(chooseBestFeature(dataSet)))
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))
    testVec = [1,0]
    result = classify(myTree, featLabels, testVec)
    print(result)
    print(majorityCnt(labels))
    filename = "决策树.txt"
    storeTree(myTree, filename)
    loadTree = grabTree(filename)
    print(loadTree)
    createPlot(myTree)
