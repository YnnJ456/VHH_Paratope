from pycaret.classification import *
import os


class PycaretWrapper:
    def __init__(self):
        pass

    def doLoadModel(self, loadPath):
        """

        :param path:
        :param fileNameList:
        :return:
        """
        modelList = []
        loadedModel = load_model(loadPath)
        resultModel = loadedModel.named_steps.trained_model
        modelList.append(resultModel)

        return modelList
