from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    # numpy tile()函数:tile(a,b) 将 a 重复 b 次
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    # numpy sum()函数:sum()将矩阵所有元素相加, axis=0 将每一列元素相加 压缩成一行, axis=1将每一行元素相加 压缩成一列
    squareDist = sum(sqdiff, axis = 1)
    dist = squareDist ** 0.5

    # 对距离进行排序
    sortedDistINdex = argsort(dist) ##arfsort()根据元素的值从小到大进行排序,返回下标


    classCount = {}
    for i in range(k):
        ## 对选取的K个样本所属的类别进行类别个数统计
        voteLabel = label[sortedDistINdex[i]]
        # dict.get("key", 0) + 1 选取字典key对应的值(若不存在则默认为0)然后+1
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes




if __name__ == "__main__":
    dataSet, labels = createDataSet()
    input = array([0.2, 2.3])
    K = 3
    output = classify(input, dataSet, labels, K)
    print(output)