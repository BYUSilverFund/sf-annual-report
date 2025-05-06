import polars as pl
from dotenv import load_dotenv
import os
import pandas as pd


class Database:
    def __init__(self) -> None:
        load_dotenv(override=True)

        endpoint = os.getenv("DB_ENDPOINT")
        port = os.getenv("DB_PORT")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        database = os.getenv("DB_NAME")

        self.uri = f"postgresql://{user}:{password}@{endpoint}:{port}/{database}"

    def get_dataframe(self, query: str) -> pd.DataFrame:
        return pl.read_database_uri(query=query, uri=self.uri).to_pandas()