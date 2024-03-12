import csv
import pandas as pd
import numpy as np
import torch
import clickhouse_driver
import clickhouse_connect
import numpy.typing as npt
import semchunk
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from spacy.lang.ru import Russian


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


def split_text(text: str) -> list[str]:
    text = 'Так, например, экспортеры и импортеры продукции имеют противоположные интересы с точки зрения уровня и динамики курса национальной валюты. Ослабление национальной валюты, то есть снижение ее цены по отношению к другим валютам, позволяет экспортеру за то же количество иностранной валюты получать большее количество национальной валюты. Для получения прежней выручки в национальной валюте и покрытия собственных издержек производитель может продать свой товар на внешнем рынке по более низкой цене, что облегчает его экспорт. При этом импортные товары становятся дороже и количество национальной валюты, необходимое для приобретения единицы импортного товара при неизменной цене в иностранной валюте, увеличивается. В частности, происходит удорожание импортных товаров инвестиционного назначения, что может негативно повлиять на темпы роста экономики. Укрепление национальной валюты, то есть повышение ее цены по отношению к другим валютам, при прочих равных условиях оказывает обратное влияние на возможности экспорта и импорта страны.'
    chunk_size = 128 # A low chunk size is used here for demo purposes.
    nlp = Russian()
    token_counter = lambda text: len(nlp(text)) # `token_counter` may be swapped out for any function capable of counting tokens.
    return semchunk.chunk(text, chunk_size=chunk_size, token_counter=token_counter)
