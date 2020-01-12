from abc import abstractmethod

import pandas as pd
import optuna
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold

from marmot.auto_model.config import Config
from marmot.ensemble_model import (EnsembleKernelSVR, EnsembleLinearReg,
                                   EnsembleLinearSVR, EnsembleRidge)
from marmot.util import get_logger


class AutoModel:

    ensembles = []

    def __init__(self, n_trials=1000, metric='mse',
                 silent=False, n_splits=3, n_repeats=10):

        self.n_trials = n_trials

        self.metric = metric

        self.config = Config()

        self.models = {}

        self.silent = silent

        self.logger = get_logger("model")

        self.n_splits = n_splits

        self.n_repeats = n_repeats

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

    def kfold_cv(self, model, X, y):
        observed = []
        predicted = []

        kf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
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
    def __call__(self, trial):
        raise NotImplementedError()


class AutoRegressor(AutoModel):

    models = [EnsembleRidge, EnsembleLinearReg,
              EnsembleLinearSVR, EnsembleKernelSVR]

    def __call__(self, trial):

        model_cls = trial.suggest_categorical("model_cls", self.models)

        n_models = trial.suggest_categorical("n_models", [10, 30, 50])

        scale = trial.suggest_categorical('scale', [True, False])

        n_trials = trial.suggest_categorical("n_trials", [20, 50])

        col_ratio = trial.suggest_categorical("col_ratio", [0.7, 0.8, 0.9, 1.0])

        row_ratio = trial.suggest_categorical("col_ratio", [0.5, 0.6, 0.7, 0.8])

        n_poly = trial.suggest_int('n_poly', 1, 3)

        model = model_cls(n_models=n_models, n_trials=n_trials,
                          scale=scale, metric=self.metric,
                          col_ratio=col_ratio, row_ratio=row_ratio)

        score = self.kfold_cv(model, self.X, self.y)

        return score


if __name__ == '__main__':
    from tests.support import get_df_boston
    from sklearn.model_selection import train_test_split

    X, y = get_df_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = AutoRegressor(n_trials=3, metric="r2")
    model.fit(X_train, y_train)
    #print(model.predidct(X_test))
    #print(model.score(X_test, y_test))
