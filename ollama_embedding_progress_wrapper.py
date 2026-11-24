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
    chunk_size: int = 32
    show_progress: bool = True  # Added flag to control visibility toggle

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        # Use the class flag to completely toggle the progress bar
        iterator = range(0, len(texts), self.chunk_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Embedding Batches")

        for i in iterator:
            batch = texts[i: i + self.chunk_size]
            batch_embeddings = super().embed_documents(batch)
            embeddings.extend(batch_embeddings)

        return embeddings
