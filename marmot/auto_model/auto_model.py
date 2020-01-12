from abc import ABCMeta, abstractmethod

import pandas as pd
import optuna
from sklearn.model_selection import RepeatedKFold

from marmot.auto_model.config import Config
from marmot.ensemble_model import (EnsembleKernelSVR, EnsembleLinearReg,
                                   EnsembleLinearSVR, EnsembleRidge)
from marmot.util import get_logger


class AutoModel(mataclass=ABCMeta):

    ensembles = []

    def __init__(self, n_trials=100, metric='mse', silent=False):

        self.n_trials = n_trials

        self.metric = metric

        self.config = Config()

        self.models = {}

        self.silent = silent

        self.logger = get_logger("model")

    def fit(self, X, y):
        self.X, self.y = self._input_validation(X, y)

        if self.silent:
            optuna.logging.disable_default_handler()
        self.logger.info("Start AutoModeling")

        study = optuna.create_study(direction=self.direction)

        study.optimize(self, n_trials=self.n_trials)

        self.best_trial = study.best_trial

        self.logger.info(f"Best {self.metric} score: "
                         + f"{round(self.best_trial.value, 2)}")
        self.logger.info(f"Best hyperparams: "
                         + f"{self.best_trial.params}")

    def kfold_cv(self, model):
        observed = []
        predicted = []

        kf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            model.fit(X_train, y_train)

            predicted += list(model.predict(X_test).flatten())
            observed += list(y_test.flatten())

        if self.metric == 'mse':
            score = mean_squared_error(observed, predicted)
        elif self.metric == 'r2':
            score = r2_score(observed, predicted)
        else:
            NotImplementedError('Unknown metric:', self.metric)

        return score

    @property
    def direction(self):
        return ("minimize" if self.metric == "mse"
                else "maximize" if self.metric == "r2"
                else "Invalid metric")

    def _input_validation(self, *args, **kwargs):
        if len(args) >= 1:
            X = args[0]
            if isinstance(X, pd.Series):
                X = pd.DataFrame(X).T
                X = X.values
            elif isinstance(X, pd.DataFrame):
                X = X.values

            if len(args) >= 2:
                y = args[1]
                if isinstance(y, pd.Series):
                    y = y.values.flatten()
                elif isinstance(y, pd.DataFrame):
                    y = y.values.flatten()

                return X, y
            else:
                return X
        else:
            raise TypeError("Unexpected X or y")

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


class AutoRegressor(AutoModel):

    ensemblelayer_candidates = [EnsembleRidge, EnsembleLinearReg,
                                EnsembleLinearSVR, EnsembleKernelSVR]

    def __call__(self):
        booster = trial.suggest_categorical('booster', ['gbtree'])
        alpha = trial.suggest_loguniform('alpha', 1e-8, 1.0)

        max_depth = trial.suggest_int('max_depth', 1, 9)
        eta = trial.suggest_loguniform('eta', 1e-8, 1.0)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        grow_policy = trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide'])

        model = self.model_cls(verbosity=0, booster=booster,
                               alpha=alpha, max_depth=max_depth, eta=eta,
                               gamma=gamma, grow_policy=grow_policy)

        score = self.kfold_cv(model)
        return score


if __name__ == '__main__':
    from tests.support import get_df_boston
    from sklearn.model_selection import train_test_split

    X, y = get_df_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = AutoRegressor()
    model.fit(X_train, y_train)
    #print(model.predidct(X_test))
    #print(model.score(X_test, y_test))
