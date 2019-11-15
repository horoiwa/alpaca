import pandas as pd
from sklearn.datasets import load_boston


def get_df_boston():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.DataFrame(boston.target, columns=["Price"])
    return X, y
