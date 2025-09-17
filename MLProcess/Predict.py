import json
import pandas as pd
import numpy as np


class Predict:
    def __init__(self, dataX, modelList, modelNameList):
        self.dataX = dataX
        self.predVectorList = []
        self.probVectorList = []
        self.modelList = modelList
        self.predVectorList_cutOff = None
        self.bestCutoffList = None
        self.modelNameList = modelNameList

    def doPredict(self):
        for model in self.modelList:
            probVector = model.predict_proba(self.dataX)
            self.probVectorList.append(probVector[:, 1])

        return self.probVectorList

    def loadCutoff(self, bestCutOffJsonPath):
        with open(bestCutOffJsonPath) as f:
            bestCutOffDict = json.load(f)
        probDf = pd.DataFrame(self.probVectorList).T
        probDf.columns = self.modelNameList
        bestCutoffList = []
        predArrList = []
        for modelName in self.modelNameList:
            probDf.loc[probDf[modelName] > bestCutOffDict[modelName], 'predVector_' + str(modelName)] = 1
            probDf.loc[probDf[modelName] <= bestCutOffDict[modelName], 'predVector_' + str(modelName)] = 0
            predArr = np.array(probDf['predVector_' + str(modelName)].values.tolist())
            predArrList.append(predArr)
            bestCutoffList.append(bestCutOffDict[modelName])
        self.predVectorList_cutOff = predArrList
        self.bestCutoffList = bestCutoffList

        return predArrList