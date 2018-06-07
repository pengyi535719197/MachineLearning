import numpy as np
import matplotlib.pyplot as plt


# 载入数据
def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()

    i = 0
    datingDataSet = np.zeros((len(lines), 3))
    labels = []

    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        # print(listFromLine)
        datingDataSet[i:] = listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        i += 1

    return datingDataSet, labels


# 画图分析
def plotDataSet(dataSet, labels):
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
    ax1 = plt.subplot(221)
    for i in range(len(labels)):
        if labels[i] == 1:
            p1 = ax1.scatter(dataSet[i, 0], dataSet[i, 1], s=2, c='r')
        elif labels[i] == 2:
            p2 = ax1.scatter(dataSet[i, 0], dataSet[i, 1], s=2, c='b')
        else:
            p3 = ax1.scatter(dataSet[i, 0], dataSet[i, 1], s=2, c='g')
    plt.xlabel("飞行里程数")
    plt.ylabel("玩游戏时间比")
    plt.legend([p1, p2, p3], ["不喜欢", "喜欢", "非常喜欢"], loc="best", frameon=False)
    ax2 = plt.subplot(222)
    for i in range(len(labels)):
        if labels[i] == 1:
            p1 = ax2.scatter(dataSet[i, 0], dataSet[i, 2], s=2, c='r')
        elif labels[i] == 2:
            p2 = ax2.scatter(dataSet[i, 0], dataSet[i, 2], s=2, c='b')
        else:
            p3 = ax2.scatter(dataSet[i, 0], dataSet[i, 2], s=2, c='g')
    plt.xlabel("飞行里程数")
    plt.ylabel("每周冰淇淋升数")
    ax3 = plt.subplot(223)
    for i in range(len(labels)):
        if labels[i] == 1:
            p1 = ax3.scatter(dataSet[i, 1], dataSet[i, 2], s=2, c='r')
        elif labels[i] == 2:
            p2 = ax3.scatter(dataSet[i, 1], dataSet[i, 2], s=2, c='b')
        else:
            p3 = ax3.scatter(dataSet[i, 1], dataSet[i, 2], s=2, c='g')
    plt.xlabel("玩游戏时间比")
    plt.ylabel("每周冰淇淋升数")
    plt.show()


# 归一化处理
# min-max标准化: x = (x-min)/(max-min)
def autoNorm(dataSet):
    MinVal = dataSet.min(axis=0)
    MaxVal = dataSet.max(axis=0)
    # print(MinVal, MaxVal)
    ranges = MaxVal - MinVal
    # normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(MinVal, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, MinVal, MaxVal


def classify(input, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diff = np.tile(input, (dataSetSize, 1)) - dataSet
    sqDiff = diff ** 2
    squareDiff = np.sum(sqDiff, axis=1)
    dist = squareDiff ** 0.5
    sortDist = np.argsort(dist)
    classCount = {}
    for i in range(k):
        className = labels[sortDist[i]]
        classCount[className] = classCount.get(className, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    return classes


def datingClassTest(dataSet, labels, k):
    Ratio = 0.1
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, Minval, MaxVal= autoNorm(dataSet)
    m = normMat.shape[0]
    numTestVecs = int(m * Ratio)
    errorCount = 0
    for i in range(numTestVecs):
        classiierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], k)
        if (classiierResult != labels[i]): errorCount += 1
    print("预测正确率为%f" % (1 - errorCount / float(numTestVecs)))


def classifyPerson(dataSet, labels, MinVal, MaxVal, k):
    resultList = ["不喜欢", "喜欢", "非常喜欢"]
    flyMiles = float(input("请输入飞行里程"))
    percOfVedioGames = float(input("请输入游戏时间占比"))
    iceCream = float(input("请输入每周吃冰淇淋升数"))
    person = [flyMiles, percOfVedioGames, iceCream]
    person = (person - MinVal) / (MaxVal - MinVal)

    ans = classify(person, dataSet, labels, k)
    ans = resultList[ans - 1]
    print(ans)


if __name__ == "__main__":
    k = 3
    DataSet, labels = file2matrix("datingTestSet2.txt")
    normDataSet, MinVal, MaxVal= autoNorm(DataSet)
    classifyPerson(normDataSet, labels, MinVal, MaxVal, k)
    datingClassTest(DataSet, labels, k)