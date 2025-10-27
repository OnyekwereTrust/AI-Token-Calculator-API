import tiktoken
from typing import Tuple
from .base import BaseTokenizer


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer using tiktoken."""
    
    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        try:
            # Map encoding names to actual tiktoken encodings
            encoding_map = {
                "o200k_base": "o200k_base",
                "cl100k_base": "cl100k_base"
            }
            actual_encoding = encoding_map.get(encoding_name, "cl100k_base")
            self.encoding = tiktoken.get_encoding(actual_encoding)
        except Exception as e:
            # Fallback to cl100k_base if the specific encoding is not available
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e2:
                raise ValueError(f"Failed to load tiktoken encoding '{encoding_name}': {e}")
    
    def count_tokens(self, text: str) -> Tuple[int, bool]:
        """Count tokens using tiktoken (exact)."""
        if not text:
            return 0, False
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens), False  # tiktoken is exact
        except Exception as e:
            raise ValueError(f"Tokenization failed: {e}")
