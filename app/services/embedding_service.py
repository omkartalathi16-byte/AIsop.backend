from sentence_transformers import SentenceTransformer
import torch


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, texts: list[str]):
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def generate_query_embedding(self, query: str):
        embedding = self.model.encode([query], convert_to_tensor=False)
        return embedding[0].tolist()
