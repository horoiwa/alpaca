import os
import json

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

    def __init__(self, config=None):
        self.config = config if config else AdapterConfig()

    def fit(self, X, y):
        X, y = self._input_validation(X, y)

    def save(self, path_to_save=None):
        """Dump config into json

        Parameters
        ----------
        path_to_save : str
            path_to_save json, ex. "~/adapter_config.json"
        """
        path_to_save = path_to_save if path_to_save else "adapter_config.json"
        if os.path.exists(path_to_save):
            print(f"Overwritten Warning: {path_to_save} already exists")
            os.remove(path_to_save)
        with open(path_to_save, "w") as f:
            config_json = self.config.to_json(indent=4, ensure_ascii=False)
            config_json = json.loads(config_json)
            json.dump(config_json, f)

    def RawToML(self, X):
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
    from tests.support import get_df_boston
    X, y = get_df_boston()
    adapter = DataAdapter()
    adapter.fit(X, y)

    X_ml = adapter.RawToML(X)
    X_raw = adapeter.MLToRaw(X_ml)
    X_ga = adapeter.RawToGA(X)
    X_raw = adapeter.GAToRaw(X_ga)
    X_ga = adapeter.MLToGA(X_ml)
    X_ml = adapeter.GAToML(X_ga)

    adapter.save("adapter_config.json")



