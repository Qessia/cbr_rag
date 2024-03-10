import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()
HF_APIKEY = os.getenv('HF_APIKEY')


class RAG:
    def __init__(self):
        self.collection = self.create_collection()
    
    def create_collection(self):
        huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=HF_APIKEY,
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device='cuda'
        )
        client = chromadb.PersistentClient(path="./database")
        try:
            collection = client.get_or_create_collection(name="test", embedding_function=huggingface_ef)
        except Exception as e:
            print('Failed to create collection')
            exit(0)
        return collection


def main():
    rag = RAG()


if __name__ == '__main__':
    main()
    