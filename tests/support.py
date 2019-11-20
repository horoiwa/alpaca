import random

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston


def get_df_boston():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.DataFrame(boston.target, columns=["Price"])
    return X, y


def get_df_boston2():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    X["Region"] = [random.choice(["Kyoto", "Austin", "Sapporo"])
                   for _ in range(X.shape[0])]

    X["invalid"] = [np.nan if i%30 != 0 else i for i in range(X.shape[0])]

    X["manynan"] = [i if i%10 != 0 else np.nan for i in range(X.shape[0])]

    X["Temperature"] = [random.choice(list(range(0, 50, 5)))
                        for _ in range(X.shape[0])]
    y = pd.DataFrame(boston.target, columns=["Price"])
    y["Tax"] = X["TAX"]
    X = X.drop(["TAX"], 1)
    return X, y
