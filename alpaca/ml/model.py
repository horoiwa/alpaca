from abc import abstractmethod

import numpy as np
import pandas as pd

from alpaca.ml.config import BaseModelConfig
from alpaca.ml.ensemble_model import (EnsembleKernelSVR, EnsembleLinearSVR,
                                      EnsembleRidge)


class BaseModel:

    def __init__(self, config=None):
        self.config = config if config else BaseModelConfig()

    def fit(self, X, y):
        X, y = self._input_validation(X, y)
        print(X, y)

    def predict(self, X, uncertainty=False):
        X = self._input_validation(X)
        pass

    def predict_proba(self):
        pass

    def score(self):
        X, y = self._input_validation(X, y)
        pass

    def _input_validation(self, *args, **kwargs):
        if len(args) >= 1:
            X = args[0]
            if isinstance(X, pd.Series):
                X = pd.DataFrame(X).T
            assert isinstance(X, pd.DataFrame), 'X must be DataFrame'

            if len(args) >= 2:
                y = args[1]
                if isinstance(y, pd.Series):
                    y = pd.DataFrame(y, columns=[y.name])
                assert isinstance(y, pd.DataFrame), 'y must be DataFrame'
                return X, y
            else:
                return X
        else:
            raise Exception("Unexpected input")


class BaseAutoModel(BaseModel):

    ensemble_layers = []

    def fit(self, X, y):
        self.config = self._optimize(X, y)
        super().fit(X, y)

    def _optimize(self, X, y):
        config = BaseModelConfig()
        model = BaseModel(config=config)
        model.fit(X, y)

        best_config = None
        return best_config

    def _evaluate(self, X, y):
        raise NotImplementedError()


class AutoRegressionModel(BaseAutoModel):

    ensemble_layers = [EnsembleRidge]



if __name__ == '__main__':
    from tests.support import get_df_boston
    args = {"n_models": 3,
            "col_ratio": 0.8,
            "row_ratio": 0.8,
            "n_trials": 10,
            "metric": "mse",
            "scale": True,
            "n_jobs": 2}

    X, y = get_df_boston()
    model = BaseModel()
    model.fit(X, y)
