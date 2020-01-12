from sklearn.model_selection import train_test_split
from .support import get_df_boston


class TestAutoModels:

    def setup_method(self):
        X, y = get_df_boston()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(X, y, test_size=0.3))
        self.n_trials = 15
        self.metric = "mse"
        self.reasonable_score = 0.6

    def teardown_method(self):
        del self.X_test
        del self.X_train
        del self.y_test
        del self.y_train

    def test_autoregressor(self):
        pass
