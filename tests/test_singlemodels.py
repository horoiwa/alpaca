import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from alpaca.ml.single_model import (DartRegCV, GBTRegCV, KernelSVRCV, LassoCV,
                                    LinearSVRCV, RidgeCV, ElasticNetCV)
from .support import get_df_boston


class TestSingelModels:
    """single modelsのテスト
    　　ボストンなのでそこまでテストスコアが悪くないことを確かめる
    """

    def setup_method(self):
        X, y = get_df_boston()
        X_sc = pd.DataFrame(StandardScaler().fit_transform(X),
                            columns=X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(X_sc, y, test_size=0.3))
        self.n_trials = 10
        self.metric = "mse"
        self.reasonable_score = 0.3

    def teardown_method(self):
        del self.X_test
        del self.X_train
        del self.y_test
        del self.y_train

    def test_params(self):
        model = RidgeCV(n_trials=self.n_trials, metric=self.metric)

        assert model.n_trials == self.n_trials
        assert model.metric == self.metric

    def test_ridge(self):
        model = RidgeCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_lasso(self):
        model = LassoCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_elasticnet(self):
        model = ElasticNetCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_linearSVR(self):
        model = LinearSVRCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_kernelSVR(self):
        model = KernelSVRCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_gbtreg(self):
        model = GBTRegCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score

    def test_dartreg(self):
        model = DartRegCV(n_trials=self.n_trials, metric=self.metric)
        model.fit(self.X_train, self.y_train)
        model.predict(self.X_test)
        score = model.score(self.X_test, self.y_test)

        assert score > self.reasonable_score
