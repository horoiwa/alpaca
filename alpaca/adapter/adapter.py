import pandas as pd
from alpaca.ml.config import AdapterConfig


class DataAdapter:

    def __init__(self, config=None):
        self.config = config if config else AdapterConfig()

    def fit(self, X, y):
        pass

    def save(self):
        pass

    @classmethod
    def from_json(cls, adapter_config_path):
        config = AdapterConfig.load_json(adapter_config_path)
        return cls(config)



if __name__ == '__main__':
    from tests.support import get_df_boston
    X, y = get_df_boston()
    adapter = DataAdapter()
    adapter.fit(X, y)

    X_ml = adapter.RawToML(X)
    X_raw = adapeter.MLToRaw(X_ml)

    adapter.save("adapter_config.json")



