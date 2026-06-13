import argparse
import json
import os
from typing import Optional

from embedding import SilmaEmbedding
from vector_db import VectorDB
from llm import GeminiLLM


class RAGSystem:
    """Main RAG system that combines embedding, vector DB, and LLM components."""

    def __init__(self, gemini_api_key: str, model_name: str = "silma-ai/silma-embeddding-matryoshka-0.1"):
        """Initialize the RAG system.

        Args:
            gemini_api_key: Gemini API key
            model_name: Name of the SentenceTransformer model to use
        """
        self.embedding_model = SilmaEmbedding(model_name=model_name)
        self.vector_db = VectorDB(embedding_model=self.embedding_model)
        self.llm = GeminiLLM(api_key=gemini_api_key)

    def build_index(self, jsonl_path: str, force_rebuild: bool = False) -> None:
        """Build the FAISS index from the JSONL file.

        Args:
            jsonl_path: Path to the JSONL file
            force_rebuild: Whether to force rebuilding the index
        """
        # Check if index already exists
        if os.path.exists(f"{self.vector_db.index_path}.idx") and not force_rebuild:
            print("Loading existing index...")
            self.vector_db.load_index()
        else:
            print(f"Building new index from {jsonl_path}...")
            self.vector_db.load_data(jsonl_path)
            self.vector_db.build_index()
            self.vector_db.save_index()

    def query(self, question: str, db_id: Optional[str] = None, top_k: int = 5) -> str:
        """Process a query through the RAG pipeline.

        Args:
            question: The user's question
            db_id: Optional database ID to filter results
            top_k: Number of results to return

        Returns:
            Generated answer
        """
        # Detect the language of the question
        language = self.llm.detect_language(question)
        print(f"Detected language: {language}")

        # Search for relevant chunks in the vector DB
        results = self.vector_db.search(question, db_id=db_id, top_k=top_k)

        # Generate a response using the LLM
        response = self.llm.generate_response(
            question, results, language=language)

        return response


def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description="RAG System for SQL Queries")
    parser.add_argument(
        "--data", type=str, default="AR_spider.jsonl", help="Path to JSONL data file")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the index")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to return")
    parser.add_argument("--db_id", type=str,
                        help="Filter results by database ID")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--question", type=str,
                        help="Question to answer (non-interactive mode)")
    parser.add_argument("--model", type=str, default="silma-ai/silma-embeddding-matryoshka-0.1",
                        help="Name of the SentenceTransformer model to use")

    args = parser.parse_args()

    # Load Gemini API key from environment variable or use default
    gemini_api_key = os.environ.get(
        "GEMINI_API_KEY", "AIzaSyDBUfMj2MkaW2yGH1sRr_6H5-FFFcYDrCo")

    # Initialize the RAG system
    rag_system = RAGSystem(gemini_api_key=gemini_api_key,
                           model_name=args.model)

    # Build or load the index
    rag_system.build_index(args.data, force_rebuild=args.rebuild)

    if args.interactive:
        print("Interactive mode: Type 'exit' to quit")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break

            # Optional database filter
            if not args.db_id:
                db_filter = input(
                    "Filter by database ID (optional, press Enter to skip): ")
                db_id = db_filter if db_filter else None
            else:
                db_id = args.db_id

            # Process the query
            response = rag_system.query(
                question, db_id=db_id, top_k=args.top_k)
            print("\nAnswer:")
            print(response)
    else:
        if args.question:
            response = rag_system.query(
                args.question, db_id=args.db_id, top_k=args.top_k)
            print(response)
        else:
            print("Please provide a question using --question or use --interactive mode")


if __name__ == "__main__":
    main()
