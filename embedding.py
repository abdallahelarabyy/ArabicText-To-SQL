import numpy as np #type: ignore
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer #type: ignore
import time
import random


class SilmaEmbedding:
    """Class to handle embeddings using the Silma Matryoshka model with SentenceTransformer."""

    def __init__(self, model_name: str = "silma-ai/silma-embeddding-matryoshka-0.1"):
        """Initialize the embedding model.

        Args:
            model_name: The name or path of the model to load
        """
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str, max_retries: int = 5, initial_delay: float = 1.0) -> np.ndarray:
        """Get embedding for a single text string.

        Args:
            text: The text to embed
            max_retries: Maximum number of times to retry on failure (kept for compatibility)
            initial_delay: Initial delay between retries (kept for compatibility)

        Returns:
            A numpy array of the embedding vector
        """
        try:
            # SentenceTransformer encode method returns embeddings directly
            return self.model.encode(text)
        except Exception as e:
            print(f"Error encoding text: {e}")
            raise

    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays with embedding vectors
        """
        try:
            # SentenceTransformer can handle batches efficiently
            embeddings = self.model.encode(texts)
            return [embedding for embedding in embeddings]
        except Exception as e:
            print(f"Error encoding batch: {e}")
            # Fall back to processing individually if batch fails
            return [self.get_embedding(text) for text in texts]

    def prepare_chunk_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """Prepare a chunk from AR_spider.jsonl for embedding.

        Args:
            chunk: A dictionary containing question, query, arabic, etc.

        Returns:
            Formatted text ready for embedding
        """
        # Combine English and Arabic question with the SQL query for better context
        formatted_text = (
            f"Question: {chunk.get('question', '')}\n"
            f"Arabic: {chunk.get('arabic', '')}\n"
            f"Query: {chunk.get('query', '')}\n"
            f"Database: {chunk.get('db_id', '')}"
        )
        return formatted_text
