import csv
import pandas as pd
import numpy as np
import torch
import clickhouse_driver
import clickhouse_connect
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


class Database:
    def __init__(self):
        """Inits ClickHouse connection and creates table"""
        self.client = clickhouse_driver.Client(host='localhost')
        self._create_db()

    def _create_db(self):
        """Creates table for embeddings"""
        self.client.execute(
            """
            CREATE TABLE embeddings (
                id UInt32,
                embedding Array(Float32),
                link String
            )
            ENGINE = MergeTree
            PRIMARY KEY (id)
            """
            )
        
    def show_data(self):
        """Prints table content"""
        print(self.client.execute("SELECT * FROM embeddings"))

    def insert(self, data: list[int, npt.NDArray[np.float32], str]):
        """Inserts data to database

        Args:
            data (list[int, npt.NDArray[np.float32], str]): data
        """
        self.client.insert('embeddings', data, column_names=['id', 'embedding', 'link'])

    def __del__(self):
        self.client.execute("DROP TABLE embeddings")
