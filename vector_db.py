import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from embedding import SilmaEmbedding


class VectorDB:
    """Class to handle FAISS vector database operations."""

    def __init__(self, embedding_model: SilmaEmbedding, index_path: str = "indexes/faiss_index"):
        """Initialize the FAISS vector database.

        Args:
            embedding_model: The embedding model to use
            index_path: Path to save/load the FAISS index
        """
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.index = None
        self.chunks = []
        self.db_indices = {}  # Maps db_id to list of indices in the FAISS index

    def load_data(self, jsonl_path: str) -> None:
        """Load data from a JSONL file.

        Args:
            jsonl_path: Path to the JSONL file
        """
        self.chunks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    self.chunks.append(chunk)
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line}")

        print(f"Loaded {len(self.chunks)} chunks from {jsonl_path}")

    def build_index(self) -> None:
        """Build the FAISS index from loaded chunks."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_data first.")

        # Group chunks by db_id
        db_chunks = {}
        for i, chunk in enumerate(self.chunks):
            db_id = chunk.get('db_id', 'unknown')
            if db_id not in db_chunks:
                db_chunks[db_id] = []
                self.db_indices[db_id] = []

            db_chunks[db_id].append((i, chunk))

        # Get embeddings for all chunks
        all_embeddings = []
        for db_id, chunks_list in db_chunks.items():
            print(f"Processing {len(chunks_list)} chunks for database {db_id}")

            for i, (chunk_idx, chunk) in enumerate(chunks_list):
                formatted_text = self.embedding_model.prepare_chunk_for_embedding(
                    chunk)
                embedding = self.embedding_model.get_embedding(formatted_text)
                all_embeddings.append(embedding)
                self.db_indices[db_id].append(len(all_embeddings) - 1)

                if i % 10 == 0 and i > 0:
                    print(
                        f"Processed {i}/{len(chunks_list)} chunks for {db_id}")

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        print(
            f"Built FAISS index with {self.index.ntotal} vectors of dimension {dimension}")

    def save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call build_index first.")

        # Create directory if it doesn't exist
        dirname = os.path.dirname(self.index_path)
        if dirname:  # Only try to create directory if there is a directory component
            os.makedirs(dirname, exist_ok=True)
        else:
            # If no directory is specified, ensure we save to a default location
            os.makedirs("indexes", exist_ok=True)
            if self.index_path == "faiss_index":  # If using the old default
                self.index_path = "indexes/faiss_index"

        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.idx")

        # Save metadata (chunks and db_indices)
        with open(f"{self.index_path}_meta.json", 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'db_indices': self.db_indices
            }, f, ensure_ascii=False)

        print(
            f"Saved index to {self.index_path}.idx and metadata to {self.index_path}_meta.json")

    def load_index(self) -> None:
        """Load the FAISS index from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{self.index_path}.idx")

        # Load metadata
        with open(f"{self.index_path}_meta.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.chunks = metadata['chunks']
            self.db_indices = metadata['db_indices']

        print(
            f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")

    def search(self, query: str, db_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for similar chunks.

        Args:
            query: The query text
            db_id: Optional database ID to filter results
            top_k: Number of results to return

        Returns:
            List of most similar chunks with similarity scores
        """
        if self.index is None:
            raise ValueError(
                "No index to search. Call build_index or load_index first.")

        # Get query embedding
        query_embedding = self.embedding_model.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')

        # If db_id is provided, search only within that database's chunks
        if db_id and db_id in self.db_indices:
            # Create a specialized index with only vectors from this db_id
            indices = self.db_indices[db_id]
            db_vectors = np.vstack([self.index.reconstruct(i)
                                   for i in indices]).astype('float32')

            temp_index = faiss.IndexFlatL2(db_vectors.shape[1])
            temp_index.add(db_vectors)

            distances, i_indices = temp_index.search(
                query_embedding, min(top_k, len(indices)))

            # Map back to original indices
            result_indices = [indices[i] for i in i_indices[0]]
        else:
            # Search the entire index
            distances, result_indices = self.index.search(
                query_embedding, top_k)
            result_indices = result_indices[0]
            distances = distances[0]

        # Get the actual chunks
        results = []
        for i, idx in enumerate(result_indices):
            if idx < len(self.chunks):  # Safety check
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(distances[i]) if i < len(distances) else 0.0
                })

        return results
