from typing import Dict, Tuple
from .base import BaseTokenizer
from .openai_tiktoken import OpenAITokenizer
from .anthropic_approx import AnthropicTokenizer
from .llama_bpe_approx import LlamaTokenizer


class TokenizerFactory:
    """Factory for creating tokenizers."""
    
    def __init__(self):
        self._tokenizers: Dict[str, BaseTokenizer] = {}
    
    def get_tokenizer(self, tokenizer_name: str) -> BaseTokenizer:
        """Get or create a tokenizer by name."""
        if tokenizer_name in self._tokenizers:
            return self._tokenizers[tokenizer_name]
        
        tokenizer = self._create_tokenizer(tokenizer_name)
        self._tokenizers[tokenizer_name] = tokenizer
        return tokenizer
    
    def _create_tokenizer(self, tokenizer_name: str) -> BaseTokenizer:
        """Create a tokenizer based on the name."""
        if tokenizer_name.startswith("o") and tokenizer_name.endswith("_base"):
            # OpenAI tiktoken encodings
            return OpenAITokenizer(tokenizer_name)
        elif tokenizer_name == "cl100k_base":
            # OpenAI GPT-4 tokenizer
            return OpenAITokenizer(tokenizer_name)
        elif tokenizer_name == "anthropic_approx_bpe":
            return AnthropicTokenizer()
        elif tokenizer_name == "llama_approx_bpe":
            return LlamaTokenizer()
        else:
            # Default to cl100k_base for unknown types
            return OpenAITokenizer("cl100k_base")
    
    def count_tokens(self, text: str, tokenizer_name: str) -> Tuple[int, bool]:
        """Count tokens using the specified tokenizer."""
        tokenizer = self.get_tokenizer(tokenizer_name)
        return tokenizer.count_tokens(text)


# Global factory instance
tokenizer_factory = TokenizerFactory()
