import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from marmot.ensemble_model import (EnsembleDartReg, EnsembleGBTReg,
                                   EnsembleKernelReg, EnsembleKernelRidge,
                                   EnsembleKernelSVR, EnsembleLinearSVR,
                                   EnsemblePLSR, EnsembleRidge)

from .support import get_df_boston


class TestEnsemblelModels:
    """single modelsのテスト
    　　ボストンなのでそこまでテストスコアが悪くないことを確かめる
    """

    def setup_method(self):
        X, y = get_df_boston()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(X, y, test_size=0.3))

        self.args = {"n_models": 10,
                     "col_ratio": 0.8,
                     "row_ratio": 0.8,
                     "n_trials": 15,
                     "metric": "mse",
                     "scale": True,
                     "scale_y": True,
                     "n_jobs": 1}

        self.reasonable_score = 0.5

    def teardown_method(self):
        del self.X_test
        del self.X_train
        del self.y_test
        del self.y_train

    def test_ensemble_ridge(self):
        model = EnsembleRidge(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_ensemble_linearsvr(self):
        model = EnsembleLinearSVR(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_ensemble_kernelsvr(self):
        model = EnsembleKernelSVR(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score
    """
    def test_ensemble_dartreg(self):
        model = EnsembleDartReg(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score
    """

    def test_ensemble_gbtreg(self):
        model = EnsembleGBTReg(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_ensemble_plsr(self):
        model = EnsemblePLSR(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_ensemble_kernelridge(self):
        model = EnsembleKernelRidge(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_ensemble_kernelreg(self):
        model = EnsembleKernelReg(**self.args)

        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score
