import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents= []

    def add(self, embeddings, documents):
        embeddings = np.asarray(embeddings, dtype= np.float32)

        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and Documents must match")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(embeddings)
        self.documents.extend(documents)


    def search(self, query_embedding, top_k=3):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not self.documents:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1,-1)

        if query_embedding.shape[1] != self.index.d:
            raise ValueError("Query dimension mismatch")

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        return[(self.documents[i], float(score)) for score, i in zip(scores[0], indices[0]) if i >=0]
    