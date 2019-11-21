import os
import json
import logging

import numpy as np
import pandas as pd
from alpaca.adapter.adapter_config import AdapterConfig


class DataAdapter:
    """
       The role of this class is to convert between the raw dataframe with
       the dataframe for machine learning and the dataframe for GA optimization

        For example, Genetic algorithms require that each columns
        in the dataframe be independent.
        On the otherhands, Machine learning requries
        a categorical variable in the dataframe should be translated
        into one-hotted columns.

        Because it is bad idea to run both machine learning modeling and
        genetic algorithm optimization with the same dataframe,
        this class mediate data exchange between the two classes,
        alpaca.ml.Model and alpaca.optimize.Optimizer
    """

    #: if unique elements in a column less than this,
    #: tha column be regarded as categorical variable
    affordable_defeat = 0.3

    def __init__(self, config=None):
        self.config = config if config else AdapterConfig()

    def fit(self, X, y):
        """generate_config

        Parameters
        ----------
        X : pd.DataFrame
            explainers
        y : pd.DataFrame
            objectives
        """
        X, y = self._input_validation(X, y)

        if X.shape[0] < 150:
            self.categorical_threshold = 15
        elif X.shape[0] < 500:
            self.categorical_threshold = 30
        else:
            self.categorical_threshold = 50

        self.mapping_variables(X, y)

    def mapping_variables(self, X, y):

        explainers = []
        explainers_type = {}
        for col in X.columns:
            explainers.append(col)

            defeat = round(X[col].isna().mean(), 3)
            if defeat > self.affordable_defeat:
                logging.warning(f'"{col}" {defeat*100}% of value is nan')
                variable_type = "no_use"
            else:
                variable_type = self._get_variabletype(X[col])

            explainers_type[col] = variable_type
            logging.info(f"{col} (explainer): {variable_type}")

        self.config.explainers = explainers
        self.config.explainers_type = explainers_type

        objective_type = {}
        for col in y.columns:
            variable_type = self._get_variabletype(y[col])
            objective_type[col] = variable_type
            logging.info(f"{col} (objective): {variable_type}")
        self.config.objectives = list(y.columns)
        self.config.objective_type = objective_type

    def _get_variabletype(self, x):
        values = x.dropna()

        n_uniques = len(set(values))
        if n_uniques <= self.categorical_threshold:
            if np.all([isinstance(val, str) for val in values]):
                return "categorical_label"
            elif np.all([isinstance(val, (int, float)) for val in values]):
                return "categorical_order"
            else:
                raise Exception("Unexpected error #11")

        try:
            values = [float(value) for value in values]
        except ValueError:
            raise Exception("Categorical variable coantains"
                            + " too many unique elements, "
                            + "Please manually set categorical threshold value")

        if np.all([value.is_integer for value in values]):
            return "numerical_int"
        else:
            return "numerical_float"

    def save(self, path_to_save):
        """Dump config into json

        Parameters
        ----------
        path_to_save : str
            path_to_save json
        """
        if os.path.exists(path_to_save):
            print(f"Overwritten Warning: {path_to_save} already exists")
            os.remove(path_to_save)
        with open(path_to_save, "w") as f:
            config_json = self.config.to_json(indent=4, ensure_ascii=False)
            config_json = json.loads(config_json)
            json.dump(config_json, f)

    def RawToML(self, X_raw):
        if X_raw.columns != self.config.explainers:
            raise Exception("Inconsitent columns")
        pass

    def RawToGA(self, X):
        pass

    def MLToRaw(self, X_ml):
        pass

    def MLtoGA(self, X_ml):
        self.RawToGA(self.MLToRaw(X_ml))

    def GAToRaw(self, X_ga):
        pass

    def GAToML(self, X_ga):
        return self.RawToML(self.GAToRaw(X_ga))

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

    def reload_config(self, path_to_config):
        with open(path_to_config, "r") as f:
            self.config = AdapterConfig.from_json(f.read())

    @classmethod
    def from_json(cls, path_to_config):
        """Load config and instanciate DataAdapter from json file

        Parameters
        ----------
        path_to_config : str
            path to json file

        Returns
        -------
        DataAdapter

        """
        with open(path_to_config, "r") as f:
            config = AdapterConfig.from_json(f.read())
        return cls(config)



if __name__ == '__main__':
    from tests.support import get_df_boston2

    logging.basicConfig(level=logging.INFO)
    logging.info('Started')
    X, y = get_df_boston2()
    adapter = DataAdapter()
    adapter.fit(X, y)
    adapter.save("example/config.json")
    logging.info('Finished')
    """
    X_ml = adapter.RawToML(X)
    X_raw = adapeter.MLToRaw(X_ml)
    X_ga = adapeter.RawToGA(X)
    X_raw = adapeter.GAToRaw(X_ga)
    X_ga = adapeter.MLToGA(X_ml)
    X_ml = adapeter.GAToML(X_ga)

    adapter.save("adapter_config.json")
    """



