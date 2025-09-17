import pandas as pd


class PackageI2Feature:
    def __init__(self, seqDict, featureDict):
        self.start = featureDict['Usage']
        aaProperty = {'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                      'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
                      'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
                      'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
                      'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
                      'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
                      'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
                      'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
                      'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
                      'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
                      'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
                      'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
                      'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                      'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
                      'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
                      'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
                      'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
                      'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
                      'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
                      'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
                      'U': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                      '-': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                      }
        aaList = 'ACDEFGHIKLMNPQRSTVWY'
        AAI2list = []
        seqList = list(seqDict.values())
        seqNameList = list(seqDict.keys())
        for w in range(len(seqList[0])):
            propList = ['property' + str(i) + '_' + str(w + 1) for i in range(1, 8)]
            for i in range(len(aaList)):
                AAI2list.append(aaList[i] + '_aa' + str(w + 1))
            AAI2list = AAI2list + propList
        i2FeatureDict = {}
        for seq, seqName in zip(seqList, seqNameList):
            i2FeatureList = []
            for aa in seq:
                aaDict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0,
                          'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
                if aa != '-':
                    aaDict[f'{aa}'] += 1
                i2FeatureList.extend(list(aaDict.values()))
                i2FeatureList.extend(aaProperty[f'{aa}'])
            i2FeatureDict[seqName] = i2FeatureList

        self.i2DFeatureDf = pd.DataFrame(i2FeatureDict).T
        self.i2DFeatureDf.columns = AAI2list

    def getOutputDf(self):
        if self.start is True:
            return self.i2DFeatureDf
