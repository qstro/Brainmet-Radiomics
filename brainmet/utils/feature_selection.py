from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import (
    f_regression,
    SelectKBest,
    f_classif,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression


class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Implementation of an MRMR feature selector for use
    with sklearn.SelectFromModel
    """

    def __init__(self, number):
        self.number = number
        self.selected_features_ = []

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        selected_feature_indices = self.MRMR(X_df, y, self.number)
        self.selected_features_ = selected_feature_indices
        return self

    def transform(self, X):
        if not self.selected_features_:
            raise RuntimeError("The fit method must be called before transform.")

        X_reduced = X[:, self.selected_features_]
        return X_reduced

    def MRMR(self, X, Y, number):
        labels = np.ravel(Y)

        encoder = LabelEncoder()
        Y = encoder.fit_transform(labels)

        min_max_scaler = MinMaxScaler()
        X_norm = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

        F = pd.Series(f_regression(X_norm, Y)[0], index=X_norm.columns)
        corr = pd.DataFrame(0.00001, index=X_norm.columns, columns=X_norm.columns)

        selected = []
        not_selected = X_norm.columns.to_list()

        for i in range(number):
            if i > 0:
                last_selected = selected[-1]
                corr.loc[not_selected, last_selected] = (
                    X_norm[not_selected]
                    .corrwith(X_norm[last_selected])
                    .abs()
                    .clip(0.00001)
                )

            score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(
                axis=1
            ).fillna(0.00001)
            best = score.index[score.argmax()]
            selected.append(best)
            not_selected.remove(best)

        selected_indices = [X.columns.get_loc(feature) for feature in selected]
        return selected_indices


def LASSO_selection(X, Y, number):
    feature_names = X.columns

    # Initialize and fit LogisticRegression with L1 (=LASSO) penalty
    lr = LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=1000, random_state=1
    )
    lr.fit(X, Y)

    selector = SelectFromModel(lr, prefit=True, max_features=number)

    selected_features_mask = selector.get_support()

    selected_feature_names = np.array(feature_names)[selected_features_mask]

    return selected_feature_names