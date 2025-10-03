# knowledge_base_manager.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import KNOWLEDGE_BASE_DIR, EMBEDDING_MODEL_NAME

class KnowledgeBaseManager:
    def __init__(self):
        print("Setting up the Knowledge Base...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.knowledge_chunks = self._load_knowledge()
        if not self.knowledge_chunks:
            raise ValueError("Knowledge base is empty.")
        self.index = self._create_faiss_index()
        print("✅ Knowledge Base is ready!")

    def _load_knowledge(self) -> list[str]:
        chunks = []
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        file_content = [line.strip() for line in f.read().splitlines() if line.strip()]
                        chunks.extend(file_content)
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
        return chunks

    def _create_faiss_index(self):
        embeddings = self.embedding_model.encode(self.knowledge_chunks, convert_to_tensor=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return index

    def search(self, query: str, k: int = 1) -> str:
        query_embedding = self.embedding_model.encode([query])
        _, I = self.index.search(np.array(query_embedding).astype('float32'), k=k)
        return self.knowledge_chunks[I[0][0]]

kb_manager = KnowledgeBaseManager()
