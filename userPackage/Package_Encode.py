from devPackage.PackageModelAmp import EncodeModelAmp
from devPackage.PackageiFeature import iFeature
from devPackage.PackagePFeature import PFeature
from devPackage.OVP import OVP
from devPackage.PackageCenterGDP import centerGDP
from devPackage.PackageBoruta import BorutaPackage
from devPackage.Normalization import Normalization
from devPackage.Package_I2 import PackageI2Feature
import pandas as pd
import random
import pickle
import os
from MLProcess.PycaretWrapper import PycaretWrapper


class EncodeAllFeatures:
    def __init__(self):
        self.featureDict = None

    def dataEncodeSetup(self, loadPklPath=None):
        path = loadPklPath
        with open(path, 'rb') as f:
            self.featureDict = pickle.load(f)

    def dataEncodeOutPut(self, dataDict, minusDict):
        i2Obj = PackageI2Feature(minusDict, self.featureDict['i2'])
        eifObj = iFeature(dataDict, self.featureDict['iFeature'])
        epfObj = PFeature(dataDict, self.featureDict['pFeature'])
        emaObj = EncodeModelAmp(dataDict, self.featureDict['ampFeature'])
        eovpObj = OVP(dataDict, self.featureDict['ovpFeature'])
        eigObj = centerGDP(minusDict, self.featureDict['centerGDPFeature'])
        i2 = i2Obj.getOutputDf()
        a = eifObj.getOutputDf()
        b = epfObj.getOutputDf()
        c = emaObj.getOutputDf()
        d = eovpObj.getOutputDf()
        g = eigObj.getOutputDf()
        encodedDf = pd.concat([i2, a], axis=1)
        encodedDf = pd.concat([encodedDf, b], axis=1)
        encodedDf = pd.concat([encodedDf, c], axis=1)
        encodedDf = pd.concat([encodedDf, d], axis=1)
        outputDf = pd.concat([encodedDf, g], axis=1)
        return outputDf

    @staticmethod
    def dataNormalization(encodIndpDf=None, loadNmlzScalerPklPath='./data/'):
        nmlzObj = Normalization(testDf=encodIndpDf)
        indpNmlzedDf = nmlzObj.robustTest(loadNmlzParamsPklPath=loadNmlzScalerPklPath)

        return indpNmlzedDf

    def dataDecidedFeatureNum(self, featureNum, saveCsvPath="./data/", dataDf=None, featRankPath="./data/", borutaMethod='XGB',
                              dataName=None):
        brtObj = BorutaPackage(modelName=borutaMethod.upper(),
                               featRankPath=featRankPath)
        featureList = brtObj.numberRanks(featureNum)
        indpCSV = dataDf[featureList]
        indpCSV.to_csv(saveCsvPath + f"/{dataName}_test_F" + str(featureNum) + ".csv")
