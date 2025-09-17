import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from boruta import BorutaPy
import xgboost as xgb
import lightgbm as lgb
import os


class BorutaPackage:
    def __init__(self, modelName="RF", featRankPath=None):
        if modelName == "XGB":
            self.model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        if modelName == "RF":
            self.model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
        if modelName == "LGB":
            self.model = lgb.LGBMClassifier(num_boost_round=100)

        self.featureAll = pd.read_csv(featRankPath, index_col=[0])
        print(self.featureAll)

    def numberRanks(self, number):
        feature_sort = self.featureAll
        feature_sort = feature_sort.reset_index(drop=True)
        df = feature_sort.iloc[0:number]
        keyList = df['feature name'].tolist()
        return keyList
