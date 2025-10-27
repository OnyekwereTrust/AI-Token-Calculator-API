
import pytest

from app.pricing_loader import PricingConfig, PricingLoader


class TestPricingLoader:
    """Test pricing loader functionality."""

    def test_pricing_config_validation(self):
        """Test pricing configuration validation."""
        # Valid config
        valid_config = {
            "vendor": "openai",
            "context": 128000,
            "input_per_1k": 0.15,
            "output_per_1k": 0.60,
            "tokenizer": "o200k_base",
            "kind": "chat",
        }

        config = PricingConfig(**valid_config)
        assert config.vendor == "openai"
        assert config.context == 128000
        assert config.input_per_1k == 0.15
        assert config.output_per_1k == 0.60
        assert config.tokenizer == "o200k_base"
        assert config.kind == "chat"

    def test_pricing_config_validation_negative_cost(self):
        """Test pricing configuration validation with negative cost."""
        invalid_config = {
            "vendor": "openai",
            "input_per_1k": -0.15,  # Negative cost
            "tokenizer": "o200k_base",
            "kind": "chat",
        }

        with pytest.raises(ValueError, match="Cost must be non-negative"):
            PricingConfig(**invalid_config)

    def test_pricing_config_embedding_model(self):
        """Test pricing configuration for embedding model."""
        embedding_config = {
            "vendor": "openai",
            "input_per_1k": 0.02,
            "tokenizer": "cl100k_base",
            "kind": "embedding",
        }

        config = PricingConfig(**embedding_config)
        assert config.vendor == "openai"
        assert config.input_per_1k == 0.02
        assert config.output_per_1k is None
        assert config.context is None
        assert config.kind == "embedding"

    def test_pricing_loader_initialization(self):
        """Test pricing loader initialization."""
        loader = PricingLoader()
        assert loader.pricing_data == {}
        assert loader.pricing_url is None

        loader_with_url = PricingLoader("https://example.com/pricing.json")
        assert loader_with_url.pricing_url == "https://example.com/pricing.json"

    def test_validate_and_store_valid_data(self):
        """Test validating and storing valid pricing data."""
        loader = PricingLoader()

        valid_data = {
            "openai:gpt-4o-mini": {
                "vendor": "openai",
                "context": 128000,
                "input_per_1k": 0.15,
                "output_per_1k": 0.60,
                "tokenizer": "o200k_base",
                "kind": "chat",
            },
            "anthropic:claude-3-5-sonnet": {
                "vendor": "anthropic",
                "context": 200000,
                "input_per_1k": 3.00,
                "output_per_1k": 15.00,
                "tokenizer": "anthropic_approx_bpe",
                "kind": "chat",
            },
        }

        loader._validate_and_store(valid_data)

        assert len(loader.pricing_data) == 2
        assert "openai:gpt-4o-mini" in loader.pricing_data
        assert "anthropic:claude-3-5-sonnet" in loader.pricing_data

        # Check individual configs
        openai_config = loader.pricing_data["openai:gpt-4o-mini"]
        assert openai_config.vendor == "openai"
        assert openai_config.context == 128000

        anthropic_config = loader.pricing_data["anthropic:claude-3-5-sonnet"]
        assert anthropic_config.vendor == "anthropic"
        assert anthropic_config.context == 200000

    def test_validate_and_store_invalid_data(self):
        """Test validating and storing invalid pricing data."""
        loader = PricingLoader()

        invalid_data = {
            "invalid:model": {
                "vendor": "openai",
                "input_per_1k": -0.15,  # Invalid negative cost
                "tokenizer": "o200k_base",
                "kind": "chat",
            },
        }

        with pytest.raises(ValueError, match="Invalid pricing config for invalid:model"):
            loader._validate_and_store(invalid_data)

    def test_get_model_config(self):
        """Test getting model configuration."""
        loader = PricingLoader()

        data = {
            "openai:gpt-4o-mini": {
                "vendor": "openai",
                "context": 128000,
                "input_per_1k": 0.15,
                "output_per_1k": 0.60,
                "tokenizer": "o200k_base",
                "kind": "chat",
            },
        }

        loader._validate_and_store(data)

        # Test existing model
        config = loader.get_model_config("openai:gpt-4o-mini")
        assert config is not None
        assert config.vendor == "openai"

        # Test non-existing model
        config = loader.get_model_config("unknown:model")
        assert config is None

    def test_list_models(self):
        """Test listing models."""
        loader = PricingLoader()

        data = {
            "openai:gpt-4o-mini": {
                "vendor": "openai",
                "context": 128000,
                "input_per_1k": 0.15,
                "output_per_1k": 0.60,
                "tokenizer": "o200k_base",
                "kind": "chat",
            },
        }

        loader._validate_and_store(data)

        models = loader.list_models()
        assert "openai:gpt-4o-mini" in models
        assert models["openai:gpt-4o-mini"]["vendor"] == "openai"
        assert models["openai:gpt-4o-mini"]["context"] == 128000
