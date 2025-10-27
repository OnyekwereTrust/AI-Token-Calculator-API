from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Base class for tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> tuple[int, bool]:
        """
        Count tokens in text.

        Returns:
            Tuple of (token_count, is_approximation)
        """
