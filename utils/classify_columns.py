import pandas as pd
def classify_columns(df, target, cat_threshold=0, text_threshold=50):
    numeric_cols, categorical_cols, text_cols = [], [], []
    for col in df.columns:
        if col in [target, "customerID"]:
            continue
        unique_vals = df[col].nunique(dropna=True)
        if pd.api.types.is_object_dtype(df[col]):
            max_len = df[col].dropna().astype(str).map(len).max()
            if max_len and max_len > text_threshold:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            categorical_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            if unique_vals <= cat_threshold:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, text_cols