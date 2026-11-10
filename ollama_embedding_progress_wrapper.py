from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
# class ProgressOllamaEmbeddings(OllamaEmbeddings):
#
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         # This mirrors how the old community version handled show_progress
#         return [self.embed_query(text) for text in tqdm(texts, desc="Embedding")]

from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm


class ProgressOllamaEmbeddings(OllamaEmbeddings):
    chunk_size: int = 32  # Adjust based on your Ollama server capacity

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        # Process in batches to leverage Ollama's native array handling
        for i in tqdm(range(0, len(texts), self.chunk_size), desc="Embedding Batches"):
            batch = texts[i: i + self.chunk_size]
            # Calls the base class embed_documents which natively handles lists
            batch_embeddings = super().embed_documents(batch)
            embeddings.extend(batch_embeddings)

        return embeddings
