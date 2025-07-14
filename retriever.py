import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.embeddings = None
        self.build_index()

    def build_index(self):
        texts = self.df.astype(str).apply(lambda x: ' '.join(x), axis=1)
        self.embeddings = self.model.encode(texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        return self.df.iloc[indices[0]]
