from dataclasses import asdict, dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:

    n_models: int = None

    col_ratio: float = 0.8

    row_ratio: float = 0.8

    n_trials: int = 10

    metric: str = None

    fillna: str = None

    poly: int = 1

    feature_selection: bool = None

    scale: bool = True

    ensemble_layer: str = None

    output: str = None


if __name__ == '__main__':
    config = Config()
    print(config.to_json())

