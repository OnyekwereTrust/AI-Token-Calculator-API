from abc import ABC, abstractmethod
from typing import Tuple


class BaseTokenizer(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> Tuple[int, bool]:
        """
        Count tokens in text.
        
        Returns:
            Tuple of (token_count, is_approximation)
        """
        pass
