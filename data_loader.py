import pandas as pd
from config import DATA_FOLDER

def load_data():
    df_client = pd.read_csv(DATA_FOLDER / "clients.csv")
    df_contracts = pd.read_csv(DATA_FOLDER / "contracts.csv")
    df_cinteractions = pd.read_csv(DATA_FOLDER / "interactions.csv")
    df_usage = pd.read_csv(DATA_FOLDER / "usage.csv")

    df_merged = df_client.merge(df_contracts, on="customerID", how="outer") \
                         .merge(df_cinteractions, on="customerID", how="outer") \
                         .merge(df_usage, on="customerID", how="outer")
    return df_client, df_contracts, df_cinteractions, df_usage, df_merged