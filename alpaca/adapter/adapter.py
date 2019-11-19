import os
import json

import pandas as pd
from alpaca.adapter.adapter_config import AdapterConfig


class DataAdapter:

    def __init__(self, config=None):
        self.config = config if config else AdapterConfig()

    def fit(self, X, y):
        X, y = self._input_validation(X, y)

    def save(self, path_to_save=None):
        path_to_save = path_to_save if path_to_save else "adapter_config.json"
        if os.path.exists(path_to_save):
            print(f"Overwritten Warning: {path_to_save} already exists")
            os.remove(path_to_save)
        with open(path_to_save, "w") as f:
            config_json = self.config.to_json(indent=4, ensure_ascii=False)
            config_json = json.loads(config_json)
            json.dump(config_json, f)

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

    adapter.save("adapter_config.json")



