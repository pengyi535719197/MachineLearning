import numpy as np
from functools import reduce

# 載入訓練好的詞條
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                  #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

'''
函數說明:根據vocalList詞彙表,將inputSet向量化, 向量的每個元素為1或者0
Parameter:
    vocabList - createVocabList返回的列表
    inputSet - 切分的詞條列表
Return:
    returnVec = 文檔向量,詞集模型
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:   # 如果詞條在詞彙表中則標記為1
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

'''
parameter
    dataSet - 整體樣本數據集
Return
    vocabSet - 返回不重複的詞條列表
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


'''
Parameters:
    trainMatrix - 訓練文檔矩陣,即setOfWords2Vec返回的returnVec構成的矩陣
    trainCategory - 訓練類別標籤,即loadDataSet返回的classVec
Returns:
    p0Vect - 非侮辱類的條件概率數組
    p1Vect - 侮辱類的條件概率數組
    pAbusive - 文檔屬於侮辱類的概率
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)     # 計算訓練文檔的數目
    numWords = len(trainMatrix[0])      #計算每個文檔的詞條數(不重复词条)
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 文檔屬於侮辱類的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

'''
parameter:
    vec2Classify - 待分類的詞條數組
    p0Vec - 侮辱類的條件概率數組
    p1Vec - 非侮辱類的條件概率數組
    pClass1 - 文檔屬於侮辱類的概率
return:
    0 - 屬於非侮辱類
    1 - 屬於侮辱類
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    print(p1)
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(classVec))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '屬於侮辱性詞彙')
    else:
        print(testEntry, '屬於非侮辱性詞彙')

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '屬於侮辱性詞彙')
    else:
        print(testEntry, '屬於非侮辱性詞彙')



if __name__ == '__main__':
    testingNB()
