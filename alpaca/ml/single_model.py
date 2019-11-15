import warnings
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.svm import SVR

warnings.filterwarnings('ignore')


class BaseSingleModelCV(metaclass=ABCMeta):

    model_cls = None
    n_splits = 3
    n_repeats = 3

    def __init__(self, n_trials=300, metric='mse'):
        self.n_trials = n_trials
        self.metric = metric

    def fit(self, X, y):
        self.X, self.y = self.check_inputs(X, y)

        study = optuna.create_study(direction='minimize')
        study.optimize(self, n_trials=self.n_trials)
        self.best_trial = study.best_trial

        print()
        print("Best score:", round(self.best_trial.value, 2))
        print("Best params:", self.best_trial.params)
        print()

        self.best_model = self.model_cls(**self.best_trial.params)
        self.best_model.fit(self.X, self.y)

    def check_inputs(self, X, y):
        assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y, columns=[y.name])
        assert isinstance(y, pd.DataFrame), 'y must be DataFrame'

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        return X, y

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X.values.reshape(1, -1))
        assert isinstance(X, pd.DataFrame), 'X must be DataFrame'

        return self.best_model.predict(X)

    def score(self, X, y):
        assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y, columns=[y.name])
        assert isinstance(y, pd.DataFrame), 'y must be DataFrame'

        return self.best_model.score(X, y)

    def valid(self, X, y, test_size=0.3):
        assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y, columns=[y.name])
        assert isinstance(y, pd.DataFrame), 'y must be DataFrame'

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        self.fit(X_train, y_train)

        print("Tran R2 score", self.score(X_train, y_train))
        print("Test R2 score", self.score(X_test, y_test))

        pred_train = self.predict(X_train)
        pred_test = self.predict(X_test)
        plt.scatter(pred_train, y_train, label='train')
        plt.scatter(pred_test, y_test, label='train')
        plt.xlabel('predict')
        plt.ylabel('observed')
        plt.show()

    def kfold_cv(self, model):
        observed = []
        predicted = []

        kf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            model.fit(X_train, y_train)

            predicted += list(model.predict(X_test).flatten())
            observed += list(y_test.values.flatten())

        if self.metric == 'mse':
            score = mean_squared_error(observed, predicted)
        elif self.metric == 'r2':
            score = r2_score(observed, predicted) * -1
        else:
            NotImplementedError('Unknown metric:', self.metric)

        return score

    @abstractmethod
    def __call__(self, trial):
        raise NotImplementedError()


class RidgeCV(BaseSingleModelCV):
    model_cls = Ridge

    def __call__(self, trial):
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e2)
        model = self.model_cls(alpha=alpha)
        score = self.kfold_cv(model)
        return score


class LassoCV(BaseSingleModelCV):
    model_cls = Lasso

    def __call__(self, trial):
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e2)
        model = self.model_cls(alpha=alpha)
        score = self.kfold_cv(model)
        return score


class LinearSVRCV(BaseSingleModelCV):
    model_cls = SVR

    def __call__(self, trial):
        C = trial.suggest_loguniform('C', 1e-2, 1e2)
        epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1e1)
        model = self.model_cls(kernel='linear', C=C,
                               epsilon=epsilon)

        score = self.kfold_cv(model)
        return score


class KernelSVRCV(BaseSingleModelCV):
    model_cls = SVR

    def __call__(self, trial):
        C = trial.suggest_loguniform('C', 1e-2, 1e2)
        epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1e1)
        gamma = trial.suggest_loguniform('gamma', 1e-3, 1e3)
        model = self.model_cls(kernel='rbf', C=C,
                               epsilon=epsilon, gamma=gamma)

        score = self.kfold_cv(model)
        return score


class GBTRegCV(BaseSingleModelCV):
    model_cls = xgb.XGBRegressor

    def __call__(self, trial):
        booster = trial.suggest_categorical('booster', ['gbtree'])
        alpha = trial.suggest_loguniform('alpha', 1e-8, 1.0)

        max_depth = trial.suggest_int('max_depth', 1, 9)
        eta = trial.suggest_loguniform('eta', 1e-8, 1.0)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        grow_policy = trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide'])

        model = self.model_cls(silent=1, booster=booster,
                               alpha=alpha, max_depth=max_depth, eta=eta,
                               gamma=gamma, grow_policy=grow_policy)

        score = self.kfold_cv(model)
        return score


class DartRegCV(BaseSingleModelCV):
    model_cls = xgb.XGBRegressor

    def __call__(self, trial):
        booster = trial.suggest_categorical('booster', ['dart'])
        alpha = trial.suggest_loguniform('alpha', 1e-8, 1.0)

        max_depth = trial.suggest_int('max_depth', 1, 16)
        eta = trial.suggest_loguniform('eta', 1e-8, 1.0)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        grow_policy = trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide'])

        sample_type = trial.suggest_categorical('sample_type',
                                                ['uniform', 'weighted'])
        normalize_type = trial.suggest_categorical('normalize_type',
                                                    ['tree', 'forest'])
        rate_drop = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        skip_drop = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        model = self.model_cls(silent=1, booster=booster,
                               alpha=alpha, max_depth=max_depth, eta=eta,
                               gamma=gamma, grow_policy=grow_policy,
                               sample_type=sample_type,
                               normalize_type=normalize_type,
                               rate_drop=rate_drop, skip_drop=skip_drop)

        score = self.kfold_cv(model)
        return score


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv('sample/boston.csv')
    y = df['Price']
    X = df.drop(['Price'], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVRCV(n_trials=300)
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))
    print(model.predict(X.iloc[1]))
    print(model.best_model)
