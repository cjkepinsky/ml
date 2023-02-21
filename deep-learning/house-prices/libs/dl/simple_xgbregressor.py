from libs.dl.simple_regressor import SimpleRegressor
from xgboost import XGBRegressor


class SimpleXGBRegressor(SimpleRegressor):
    def __init__(self, X_train, y_train, X_valid, y_valid):
        SimpleRegressor.__init__(self, X_train, y_train, X_valid, y_valid)


def build_model(self, units=7, max_depth=20, subsample=1, eta=0.3, seed=5):
    print('Building XGBRegressor model...')

    self.model = XGBRegressor(
        n_estimators=units
        , max_depth=max_depth
        , min_child_weight=1
        , subsample=subsample
        , eta=eta
        , seed=seed
        , n_jobs=4
    )
