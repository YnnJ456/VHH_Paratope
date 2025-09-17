import pickle
import pandas as pd


class Normalization:
    def __init__(self, testDf):
        if testDf is not None:
            self.testDfIndex = testDf.index.to_list()
            self.testDfFeatureCol = testDf.columns.to_list()
            self.testArray = testDf.values

    def robustTest(self, loadNmlzParamsPklPath=None):
        with open(loadNmlzParamsPklPath, 'rb') as f:
            robustSca = pickle.load(f)
            scalerDf = robustSca.fit_transform(self.testArray)

        scalerDf = pd.DataFrame(scalerDf, index=self.testDfIndex, columns=self.testDfFeatureCol)
        return scalerDf





