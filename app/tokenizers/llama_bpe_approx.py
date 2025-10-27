from typing import Tuple
from .base import BaseTokenizer


class LlamaTokenizer(BaseTokenizer):
    """Llama/Mistral tokenizer using approximation."""
    
    def __init__(self):
        # Simple approximation: ~3.5 characters per token for English text
        self.chars_per_token = 3.5
    
    def count_tokens(self, text: str) -> Tuple[int, bool]:
        """Count tokens using character-based approximation."""
        if not text:
            return 0, True
        
        # Simple approximation based on character count
        char_count = len(text)
        token_count = int(char_count / self.chars_per_token)
        
        # Ensure minimum of 1 token for non-empty text
        return max(1, token_count), True
