import pandas as pd
import numpy as np

from alpaca.ml.config import BaseModelConfig


def _input_validation(func):
    def wrapper(self, *args, **kwargs):
        if len(args) >= 1:
            X = args[0]
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

    return wrapper


class BaseModel:

    def __init__(self, config=None):
        self.config = config if config else BaseModelConfig()

    @_input_validation
    def fit(self, X, y):
        self.input_validation(X, y)

    def predict(self, uncertainty=False):
        pass

    def predict_proba(self):
        pass

    def score(self):
        pass

    @classmethod
    def from_config(cls, config):
        return BaseModel(config)


class RegressionModel(BaseModel):

    @classmethod
    def optimize(cls):
        pass


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
