import pandas as pd
import math


class TransformToFasta:
    def __init__(self, windowSize, H, fullDataCsvPath):
        # self.fullData = pd.read_excel(fullDataCsvPath, index_col=[0])
        self.fullData = pd.read_csv(fullDataCsvPath)
        self.window = int(windowSize[1:])
        self.hNum = int(H[1])
        addFrame = int((self.window / 2) - 0.5)
        dataColumn = self.fullData.columns.tolist()
        if self.hNum == 1:
            alightStart = dataColumn.index('24')  # 24-44
            alightEnd = dataColumn.index('44')
        elif self.hNum == 2:
            alightStart = dataColumn.index('47')  # 47-71
            alightEnd = dataColumn.index('71')
        elif self.hNum == 3:
            alightStart = dataColumn.index('105')  # 105-119
            alightEnd = dataColumn.index('119')
        self.realLoc = dataColumn[(alightStart - addFrame):(alightEnd + 1 + addFrame)]
        self.splitDataDf = None

    def splitData(self):
        split_H_Data = self.fullData[self.realLoc]
        split_H_Data.index = self.fullData['Id']
        dataSeries = split_H_Data.apply(lambda row: ''.join(map(str, row)), axis=1)
        self.splitDataDf = pd.DataFrame(dataSeries, columns=['data_gap'])

    def outputFasta(self, testFastaPath, testFastaGerLogPath):
        delSeqNumTimes = 0
        middle = math.floor(self.window / 2)
        returnSeqDict = {}
        logDict = {}
        with open(testFastaPath, 'w') as f:
            for seqName in self.splitDataDf.index:
                elementStr = self.splitDataDf.loc[self.splitDataDf.index == seqName, 'data_gap'][0]
                logDict[seqName] = []
                times = 0
                for i in range(len(elementStr) - self.window + 1):
                    seq = ''.join(str(v) for v in elementStr[i:i + self.window])
                    if seq[middle] == '-':
                        delSeqNumTimes += 1
                    else:
                        if seq in returnSeqDict.keys():
                            returnSeqDict[seq] += 1
                        else:
                            returnSeqDict[seq] = 1
                        if seqName == 'Antigen_80':
                            print(1)
                        print(f'>[{seqName}]_[{seq}]_[{times}]', file=f)  # *$*
                        print(seq, file=f)
                        times += 1

            returnSeqDf = pd.DataFrame(returnSeqDict.values(), index=returnSeqDict.keys(), columns=['appear times'])
            returnSeqDf.to_csv(testFastaGerLogPath)
            f.close()
