from dataclasses import asdict, dataclass

from alpaca.ml.ensemble_model import EnsembleRidge, BaseEnsembleModel
from alpaca.ml.preprocess import BaseFillna, MeanFillna, KneignborFillna
from alpaca.ml.aggregate import BaseAggregate, MeanAggregate, MedianAggregate


@dataclass
class BaseModelConfig:

    fillna: BaseFillna = MeanFillna

    poly: int = 1

    scale = True

    ensemble_layer: BaseEnsembleModel = EnsembleRidge

    aggregate: BaseAggregate = MeanAggregate

