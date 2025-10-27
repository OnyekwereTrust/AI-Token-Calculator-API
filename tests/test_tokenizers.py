from app.tokenizers import tokenizer_factory
from app.tokenizers.anthropic_approx import AnthropicTokenizer
from app.tokenizers.llama_bpe_approx import LlamaTokenizer
from app.tokenizers.openai_tiktoken import OpenAITokenizer


class TestTokenizers:
    """Test tokenizer functionality."""

    def test_openai_tokenizer_exact(self):
        """Test OpenAI tokenizer with known text."""
        tokenizer = OpenAITokenizer("o200k_base")

        # Test with empty string
        tokens, approx = tokenizer.count_tokens("")
        assert tokens == 0
        assert approx is False

        # Test with simple text
        tokens, approx = tokenizer.count_tokens("Hello world")
        assert tokens > 0
        assert approx is False

    def test_anthropic_tokenizer_approx(self):
        """Test Anthropic tokenizer approximation."""
        tokenizer = AnthropicTokenizer()

        # Test with empty string
        tokens, approx = tokenizer.count_tokens("")
        assert tokens == 0
        assert approx is True

        # Test with simple text
        tokens, approx = tokenizer.count_tokens("Hello world")
        assert tokens > 0
        assert approx is True

        # Test with longer text
        long_text = "Hello world " * 100
        tokens, approx = tokenizer.count_tokens(long_text)
        assert tokens > 0
        assert approx is True

    def test_llama_tokenizer_approx(self):
        """Test Llama tokenizer approximation."""
        tokenizer = LlamaTokenizer()

        # Test with empty string
        tokens, approx = tokenizer.count_tokens("")
        assert tokens == 0
        assert approx is True

        # Test with simple text
        tokens, approx = tokenizer.count_tokens("Hello world")
        assert tokens > 0
        assert approx is True

    def test_tokenizer_factory(self):
        """Test tokenizer factory."""
        # Test OpenAI tokenizer
        tokens, approx = tokenizer_factory.count_tokens("Hello", "o200k_base")
        assert tokens > 0
        assert approx is False

        # Test Anthropic tokenizer
        tokens, approx = tokenizer_factory.count_tokens("Hello", "anthropic_approx_bpe")
        assert tokens > 0
        assert approx is True

        # Test Llama tokenizer
        tokens, approx = tokenizer_factory.count_tokens("Hello", "llama_approx_bpe")
        assert tokens > 0
        assert approx is True

    def test_tokenizer_factory_unknown(self):
        """Test tokenizer factory with unknown tokenizer."""
        # Should default to OpenAI tokenizer
        tokens, approx = tokenizer_factory.count_tokens("Hello", "unknown_tokenizer")
        assert tokens > 0
        assert approx is False

    def test_tokenizer_consistency(self):
        """Test tokenizer consistency for same input."""
        text = "This is a test sentence for tokenization."

        # OpenAI tokenizer should be consistent
        tokens1, _ = tokenizer_factory.count_tokens(text, "o200k_base")
        tokens2, _ = tokenizer_factory.count_tokens(text, "o200k_base")
        assert tokens1 == tokens2

        # Anthropic tokenizer should be consistent
        tokens1, _ = tokenizer_factory.count_tokens(text, "anthropic_approx_bpe")
        tokens2, _ = tokenizer_factory.count_tokens(text, "anthropic_approx_bpe")
        assert tokens1 == tokens2

    def test_tokenizer_edge_cases(self):
        """Test tokenizer edge cases."""
        # Empty string
        tokens, approx = tokenizer_factory.count_tokens("", "o200k_base")
        assert tokens == 0
        assert approx is False

        # Whitespace only
        tokens, approx = tokenizer_factory.count_tokens("   ", "o200k_base")
        assert tokens >= 0

        # Special characters
        special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        tokens, approx = tokenizer_factory.count_tokens(special_text, "o200k_base")
        assert tokens > 0
        assert approx is False

        # Unicode characters
        unicode_text = "Hello 世界 🌍"
        tokens, approx = tokenizer_factory.count_tokens(unicode_text, "o200k_base")
        assert tokens > 0
        assert approx is False
