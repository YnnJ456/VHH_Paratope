import os.path
from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.Package_transformToFasta import TransformToFasta
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict


class VhhParatope_v2:
    def __init__(self, paramDict):
        self.windowSize = paramDict['windowSize']
        self.hName = paramDict['hName']
        self.featureNum = paramDict['featureNum']
        self.testDataDir = paramDict['testDataDir']
        self.outputDir = paramDict['outputDir']
        self.fullDataCsv = paramDict['fullDataCsvPath']
        self.clfName = paramDict['clfName']
        self.featureTypeDictPklPath = None
        self.nmlzScalerPklPath = None
        self.testNmlzCsvPath = None
        self.encodedMlDataDirPath = None
        self.featRankCsvPath = None
        self.loadModelPklPath = None
        self.cutoff = None
        self.fastaPath = None
        self.fastaLogPath = None

    def checkPath(self):
        # ======================================================================================================================
        self.fastaPath = f'../data/mlData/newFasta/{self.windowSize}{self.hName}_test.FASTA'
        self.fastaLogPath = f'../data/mlData/newFasta/{self.windowSize}{self.hName}_log.txt'
        # ======================================================================================================================
        self.featureTypeDictPklPath = '../data/param/featureTypeDict.pkl'
        self.cutoff = f'../data/param/{self.windowSize}{self.hName}.json'
        self.nmlzScalerPklPath = f'../data/param/{self.windowSize}{self.hName}_Scaler.pkl'  # normalize.pkl
        self.featRankCsvPath = f'../data/boruta/{self.windowSize}{self.hName}_Boruta-featureRank-XGB.csv'
        # ======================================================================================================================
        self.encodedMlDataDirPath = f'../data/mlData/newEncodedData/'
        self.testNmlzCsvPath = self.encodedMlDataDirPath + f'[{self.windowSize}{self.hName}]_test_allFeature.csv'
        self.loadModelPklPath = f'../data/model/{self.windowSize}/{self.hName}/{self.clfName}'
        # ======================================================================================================================
        print(f' ML model used {self.windowSize}_{self.hName}')
        print(f' Feature Number is {self.featureNum}')
        print('Start Check path')

        if os.path.isdir(self.encodedMlDataDirPath):
            print('encodedMlDataDirPath is exist')
        else:
            raise FileNotFoundError('ERROR!!! encodedMlDataDirPath error!!! path is not exist!!!')

        if os.path.isfile(self.featureTypeDictPklPath):
            print('featureTypeDict.pkl of featureTypeDictPklPath exist')
        else:
            raise FileNotFoundError('ERROR!!! featureTypeDictPklPath error!!! featureTypeDict.pkl is not exist!!!')

        if os.path.isfile(self.featRankCsvPath):
            print(
                f'{self.windowSize}{self.hName}_Boruta-featureRank-XGB.csv of featRankCsvPath exist')
        else:
            raise FileNotFoundError(
                f'ERROR!!! featRankCsvPath error!!! {self.windowSize}{self.hName}_Boruta-featureRank-XGB.csv is not exist!!!')

        if os.path.isfile(self.testNmlzCsvPath):
            print(f'testNmlzCsvPath already exist test.csv, new file will cover old one')
        else:
            print(f'test.csv of testNmlzCsvPath is not exist, new file will be created')

        if os.path.isfile(f'{self.loadModelPklPath}.pkl'):
            print('model of loadModelPklPath exist')
        else:
            raise FileNotFoundError(
                f'ERROR!!! loadModelPklPath error!!! /{self.windowSize}/{self.windowSize}/{self.clfName}.pkl is not exist!!!')

        if os.path.isdir(self.outputDir):
            print('outputDir exist')
        else:
            raise FileNotFoundError('ERROR!!! outputDir error!!! path not exist!!!')

    def run(self):
        # ======================================================================================================================
        transObj = TransformToFasta(windowSize=self.windowSize, H=self.hName, fullDataCsvPath=self.fullDataCsv)
        transObj.splitData()
        transObj.outputFasta(testFastaPath=self.fastaPath, testFastaGerLogPath=self.fastaLogPath)
        ldObj = LoadDataset()
        seqDict, seqMinusDict = ldObj.readVHH_Fasta(File=self.fastaPath)
        print("converted!!!")
       # ======================================================================================================================
        print("start feature encode")
        encodeObj = EncodeAllFeatures()
        encodeObj.dataEncodeSetup(loadPklPath=self.featureTypeDictPklPath)
        encodeTestDf = encodeObj.dataEncodeOutPut(dataDict=seqDict, minusDict=seqMinusDict)  # test data
        testNmlzDf = encodeObj.dataNormalization(encodIndpDf=encodeTestDf,
                                                 loadNmlzScalerPklPath=self.nmlzScalerPklPath)  # test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔
        testNmlzDf.to_csv(self.testNmlzCsvPath)  # 把全部 feature 做完 nmlz 的結果存成 csv 檔 (可檢查 feature number 以及 nmlz 的結果)
        encodeObj.dataDecidedFeatureNum(featureNum=self.featureNum, saveCsvPath=self.encodedMlDataDirPath,
                                        dataDf=testNmlzDf,
                                        featRankPath=self.featRankCsvPath,
                                        dataName=f'[{self.windowSize}{self.hName}]',
                                        borutaMethod='XGB')
        print("feature encoding done")
        # ======================================================================================================================
        pycObj = PycaretWrapper()
        model = pycObj.doLoadModel(loadPath=self.loadModelPklPath)
        dataTestDf = pd.read_csv(f'{self.encodedMlDataDirPath}/[{self.windowSize}{self.hName}]_test_F{self.featureNum}.csv', index_col=[0])
        predObjTest = Predict(dataX=dataTestDf, modelList=model, modelNameList=[f'{self.clfName}'])
        probVectorListTest = predObjTest.doPredict()
        predVectorListTest = predObjTest.loadCutoff(bestCutOffJsonPath=self.cutoff)
        predVectorDfTest = pd.DataFrame(predVectorListTest, index=[f'{self.windowSize}{self.hName}'], columns=dataTestDf.index).T
        probVectorDfTest = pd.DataFrame(probVectorListTest, index=[f'{self.windowSize}{self.hName}'], columns=dataTestDf.index).T
        predVectorDfTest.to_csv(
            self.outputDir + f'[{self.windowSize}_{self.hName}]_predVectorTest.csv')
        probVectorDfTest.to_csv(
            self.outputDir + f'[{self.windowSize}_{self.hName}]_probVectorTest.csv')
        # ======================================================================================================================

        print('All Work Done!!!')
        print('Output Prediction Result:')
        print(
            f'peptideML_VHH/data/mlScore/PredOutput/[{self.windowSize}_{self.hName}]_predVectorTest.csv')
        print(
            f'peptideML_VHH/data/mlScore/PredOutput/[{self.windowSize}_{self.hName}]_probVectorTest.csv')
