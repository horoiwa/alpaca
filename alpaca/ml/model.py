from abc import abstractmethod, ABCMeta


class BaseModel(metaclass=ABCMeta):

    def __init__(self):
        pass


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
