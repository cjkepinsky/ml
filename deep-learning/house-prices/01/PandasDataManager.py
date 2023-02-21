import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from libs.simple_processing import separate_target, split_train_test
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.decomposition import PCA

class PandasDataManager:
    def __init__(self):
        self.reduce = None
        self.rescalex = None
        self.test_df = None
        self.train_df = None
        self.train_data_from_beginning = None
        self.train_perc = None
        self.test_perc = None
        self.target_name = None
        self.X_train = None
        self.y_train = None
        self.df = None
        self.y_test = None
        self.y_pred = None

    def get_data(self):
        return self.df

    def load_split_train_valid_test(self, path, target_name
                                    , test_perc=0.01
                                    , train_perc=0.65
                                    , train_data_from_beginning=True
                                    , rescalex=True
                                    , reduce=None):
        self.df = pd.read_csv(path)
        self.target_name = target_name
        self.test_perc = test_perc
        self.train_perc = train_perc
        self.train_data_from_beginning = train_data_from_beginning
        self.rescalex = rescalex
        self.reduce = reduce

        tlength = int(len(self.df) * self.test_perc)
        length = int(len(self.df)) - tlength

        if self.train_data_from_beginning:
            self.train_df = self.df[:length]
            self.test_df = self.df[length:]
        else:
            self.train_df = self.df[tlength:]
            self.test_df = self.df[:tlength]

        X, y = separate_target(self.train_df, target_name)
        Xt, yt = separate_target(self.test_df, target_name)

        if self.rescalex:
            xs = MinMaxScaler()
            X = pd.DataFrame(xs.fit_transform(X), columns=X.columns)
            Xt = pd.DataFrame(xs.fit_transform(Xt), columns=Xt.columns)

        if self.reduce is not None:
            pca_X = PCA(n_components=self.reduce, random_state=42)
            pca_X.fit(X)
            X = pca_X.transform(X)
            pca_X.fit(Xt)
            Xt = pca_X.transform(Xt)

        X_train, X_valid, y_train, y_valid = split_train_test(X, y, self.train_perc)

        return X_train, X_valid, y_train, y_valid, Xt, yt
