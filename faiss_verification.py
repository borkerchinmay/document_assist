import numpy as np
from vector_store import VectorStore

store = VectorStore(3)

embeddings = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)

documents = ["cats", "dogs", "birds"]

store.add(embeddings, documents)

results = store.search(np.array([0.9, 0.1, 0], dtype=np.float32), top_k=2)

print(results)