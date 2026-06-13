import google.generativeai as genai  # type: ignore
from typing import List, Dict, Any, Optional


class GeminiLLM:
    """Class to handle interactions with Google's Gemini API."""

    def __init__(self, api_key: str):
        """Initialize the Gemini LLM with API key.

        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)

        # Get default model
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self,
                          query: str,
                          retrieved_chunks: List[Dict[str, Any]],
                          language: str = "english") -> str:
        """Generate a response using the Gemini API.

        Args:
            query: The user's question
            retrieved_chunks: Retrieved chunks from the vector database
            language: The language to respond in (english or arabic)

        Returns:
            Generated SQL query only
        """
        # Format the context from retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Build the prompt to return only SQL
        prompt = f"""
You are an AI assistant specialized in database queries and SQL.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Based on the above context, generate ONLY the SQL query that answers the user's question. 
Include NO explanations, comments, or additional text - JUST the raw SQL query without any markdown formatting.
"""

        # Generate a response
        response = self.model.generate_content(prompt)

        # Extract only the SQL query, removing any potential markdown formatting
        sql_query = response.text.strip()

        # Remove any markdown code blocks if present
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        return sql_query

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a context string.

        Args:
            retrieved_chunks: Retrieved chunks from the vector database

        Returns:
            Formatted context string
        """
        formatted_context = ""

        for i, result in enumerate(retrieved_chunks):
            # Check if the result has 'entry' or 'chunk' key
            chunk = result.get('entry', result.get('chunk', {}))
            score = result.get('score', 0.0)

            formatted_context += f"CHUNK {i+1} (Relevance score: {score:.4f}):\n"
            formatted_context += f"- Question (English): {chunk.get('question', 'N/A')}\n"
            formatted_context += f"- Question (Arabic): {chunk.get('arabic', 'N/A')}\n"
            formatted_context += f"- SQL Query: {chunk.get('query', 'N/A')}\n"
            formatted_context += f"- Database: {chunk.get('db_id', 'N/A')}\n\n"

        return formatted_context

    def detect_language(self, text: str) -> str:
        """Detect if the input text is in Arabic or English.

        Args:
            text: Input text

        Returns:
            'arabic' or 'english'
        """
        # Simple detection based on character codes
        # Arabic Unicode range is from U+0600 to U+06FF
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')

        # If more than 15% of characters are Arabic, consider it Arabic
        if arabic_chars / max(len(text), 1) > 0.15:
            return "arabic"
        else:
            return "english"
