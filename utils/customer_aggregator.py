from sklearn.base import BaseEstimator, TransformerMixin

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_column="customerID", target_column="Churn"):
        self.id_column = id_column
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        aggregated = X.groupby(self.id_column, as_index=False).agg({
            col: "median" if X[col].dtype.kind in "iufc" else "first"
            for col in X.columns if col != self.id_column
        })
        return aggregated
