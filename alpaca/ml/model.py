from abc import abstractmethod, ABCMeta
from dataclasses import asdict, dataclass

from alpaca.ml.ensemble_model import EnsembleRidge, BaseEnsembleModel
from alpaca.ml.preprocess import BaseFillna, KneignborFillna
from alpaca.ml.postprocess import BaseAverage, MeanAverage


@dataclass
class BaseModelConfig:

    fillna: BaseFillna = KneignborFillna

    poly: int = 1

    ensemble_layer: BaseEnsembleModel = EnsembleRidge

    output_layer: BaseAverage = MeanAverage


class BaseModel(metaclass=ABCMeta):

    def __init__(self, config=None):
        pass

    @classmethod
    def from_config(cls, config):
        return BaseModel(config)

    def add_model(self):
        pass


RegressionModel = BaseModel.from_config(BaseModelConfig())


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
    print(type(EnsembleRidge()))
    print(isinstance(EnsembleRidge(), BaseEnsembleModel))
    print(asdict(BaseModelConfig(EnsembleRidge)))
