import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def get_tenure_bin_series(df, n_bins=6):
    """
    Create tenure bins using quantiles for balanced group sizes.
    Falls back to equal-width bins if quantiles fail.
    """
    tenure_clean = pd.to_numeric(df["tenure"], errors="coerce")
    try:
        tenure_bins = pd.qcut(
            tenure_clean,
            q=n_bins,
            labels=[f"Q{i+1}" for i in range(n_bins)],
            duplicates="drop"
        )
    except ValueError:
        tenure_bins = pd.cut(
            tenure_clean,
            bins=n_bins,
            labels=[f"Bin{i+1}" for i in range(n_bins)],
            include_lowest=True
        )
    return tenure_bins

def impute_total_charges(df):
    """
    Impute missing TotalCharges using median by tenure bin.
    """
    df = df.copy()
    tenure_bins = get_tenure_bin_series(df)
    usage_mask = df["TotalCharges"].isna()

    df_temp = df.copy()
    df_temp["tenure_bin"] = tenure_bins
    group_medians = df_temp.groupby("tenure_bin", observed=True)["TotalCharges"].median()

    df["TotalCharges_imputed_flag"] = 0
    df.loc[usage_mask, "TotalCharges"] = df.loc[usage_mask].apply(
        lambda row: group_medians.get(tenure_bins[row.name]),
        axis=1
    )

    fallback_median = pd.to_numeric(df["TotalCharges"], errors="coerce").median()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(fallback_median)
    df.loc[usage_mask, "TotalCharges_imputed_flag"] = 1

    return df

def impute_avg_data_usage(df):
    """
    Impute AvgDataUsage_GB using median by ContractType, InternetService, and tenure bin.
    """
    df = df.copy()
    tenure_bins = get_tenure_bin_series(df)
    group_cols = ["ContractType", "InternetService"]
    usage_mask = df["AvgDataUsage_GB"].isna()

    df_temp = df.copy()
    df_temp["tenure_bin"] = tenure_bins
    group_medians = df_temp.groupby(group_cols + ["tenure_bin"], observed=True)["AvgDataUsage_GB"].median()

    df["AvgDataUsage_GB_imputed_flag"] = 0
    df.loc[usage_mask, "AvgDataUsage_GB"] = df.loc[usage_mask].apply(
        lambda row: group_medians.get((row["ContractType"], row["InternetService"], tenure_bins[row.name])),
        axis=1
    )

    fallback_median = df["AvgDataUsage_GB"].median()
    df["AvgDataUsage_GB"] = df["AvgDataUsage_GB"].fillna(fallback_median)
    df.loc[usage_mask, "AvgDataUsage_GB_imputed_flag"] = 1

    return df

def impute_numeric_fields(df):
    """
    Fill basic numeric fields with 0, -1, or median values.
    """
    df = df.copy()
    df["NbContacts"] = df["NbContacts"].fillna(0)
    df["LastContactDays"] = df["LastContactDays"].fillna(-1)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
    df["NumCalls"] = df["NumCalls"].fillna(df["NumCalls"].median())
    return df

def impute_satisfaction_score(df):
    """
    Impute SatisfactionScore based on whether the client was contacted.
    """
    df = df.copy()
    mask_contacted = (df["NbContacts"] > 0) & df["SatisfactionScore"].isna()
    mask_non_contacted = (df["NbContacts"] == 0) & df["SatisfactionScore"].isna()

    median_contacted = df.loc[df["NbContacts"] > 0, "SatisfactionScore"].median()
    median_non_contacted = df.loc[df["NbContacts"] == 0, "SatisfactionScore"].median()

    df["SatisfactionScore_no_response_flag"] = 0
    df.loc[mask_contacted, "SatisfactionScore"] = median_contacted
    df.loc[mask_non_contacted, "SatisfactionScore"] = median_non_contacted
    df.loc[mask_contacted | mask_non_contacted, "SatisfactionScore_no_response_flag"] = 1

    return df

def impute_feedback_text(df):
    """
    Fill missing feedback text with 'No feedback'.
    """
    df = df.copy()
    df["FeedbackText"] = df["FeedbackText"].fillna("No feedback")
    return df

def impute_categorical_fields(df):
    """
    Fill missing categorical fields with 'Unknown'.
    """
    df = df.copy()
    cat_fill = ["Partner", "Dependents", "ContractType", "PaymentMethod", "TVPackage", "TechSupport"]
    for col in cat_fill:
        df[col] = df[col].fillna("Unknown")
    return df

def impute_internet_service(df):
    """
    Impute InternetService based on presence or absence of usage evidence.
    """
    df = df.copy()
    explainable = df[
        df["InternetService"].isna() &
        df["AvgDataUsage_GB"].isna() &
        df["TechSupport"].isna() &
        df["TVPackage"].isna()
    ].index

    unjustified = df[
        df["InternetService"].isna() &
        (df["AvgDataUsage_GB"].notna() | df["TechSupport"].notna() | df["TVPackage"].notna())
    ].index

    df["InternetService_imputed_flag"] = 0
    df.loc[explainable, "InternetService"] = "Not relevant"
    df.loc[unjustified, "InternetService"] = "Unknown"
    df.loc[explainable.union(unjustified), "InternetService_imputed_flag"] = 1

    return df

def clean_columns(df):
    """
    Run all imputation steps in sequence.
    """
    df = df.copy()
    df = impute_numeric_fields(df)
    df = impute_total_charges(df)
    df = impute_avg_data_usage(df)
    df = impute_satisfaction_score(df)
    df = impute_feedback_text(df)
    df = impute_categorical_fields(df)
    df = impute_internet_service(df)
    return df

def summarize_imputations(df):
    """
    Return a summary of how many values were imputed per flag column.
    """
    flags = [col for col in df.columns if col.endswith("_imputed_flag")]
    summary = {col: int(df[col].sum()) for col in flags}
    return pd.Series(summary, name="Imputed Count")


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X = X.copy()
        return clean_columns(X)
    
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Concatène les colonnes texte si X est un DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.fillna("").astype(str)
            return X.apply(lambda row: ' '.join(row.values), axis=1)
        # Si X est une série
        return X.fillna("").astype(str)
