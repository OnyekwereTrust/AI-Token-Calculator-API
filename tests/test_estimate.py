import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.pricing_loader import PricingLoader
from app.services.estimator import EstimationService
from app.schemas import EstimateRequest, RAGConfig


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def sample_pricing():
    """Sample pricing data for testing."""
    from app.pricing_loader import PricingConfig
    
    return {
        "openai:gpt-4o-mini": PricingConfig(
            vendor="openai",
            context=128000,
            input_per_1k=0.15,
            output_per_1k=0.60,
            tokenizer="cl100k_base",
            kind="chat"
        ),
        "anthropic:claude-3-5-sonnet": PricingConfig(
            vendor="anthropic",
            context=200000,
            input_per_1k=3.00,
            output_per_1k=15.00,
            tokenizer="anthropic_approx_bpe",
            kind="chat"
        ),
        "openai:text-embedding-3-small": PricingConfig(
            vendor="openai",
            input_per_1k=0.02,
            tokenizer="cl100k_base",
            kind="embedding"
        )
    }


@pytest.fixture
def estimation_service(sample_pricing):
    """Estimation service fixture."""
    return EstimationService(sample_pricing)


class TestEstimationService:
    """Test estimation service functionality."""
    
    def test_estimate_basic(self, estimation_service):
        """Test basic estimation."""
        request = EstimateRequest(
            model="openai:gpt-4o-mini",
            system="You are helpful.",
            user="Summarize the article.",
            expected_output_tokens=250
        )
        
        result = estimation_service.estimate(request)
        
        assert result.model == "openai:gpt-4o-mini"
        assert result.input_tokens > 0
        assert result.output_tokens == 250
        assert result.cost > 0
        assert result.tokenizer.name == "cl100k_base"
        assert result.tokenizer.approx is False
    
    def test_estimate_with_rag(self, estimation_service):
        """Test estimation with RAG configuration."""
        request = EstimateRequest(
            model="openai:gpt-4o-mini",
            user="Answer the question.",
            expected_output_tokens=100,
            rag=RAGConfig(
                embedding_tokens=1000,
                num_vectors_read=5,
                vector_read_fee_per_1k=0.01
            )
        )
        
        result = estimation_service.estimate(request)
        
        assert result.cost > 0
        assert result.breakdown.embedding_cost > 0
        assert result.breakdown.vector_io_cost > 0
    
    def test_estimate_unknown_model(self, estimation_service):
        """Test estimation with unknown model."""
        request = EstimateRequest(
            model="unknown:model",
            user="Test",
            expected_output_tokens=100
        )
        
        with pytest.raises(ValueError, match="Unknown model"):
            estimation_service.estimate(request)
    
    def test_context_warning(self, estimation_service):
        """Test context utilization warning."""
        request = EstimateRequest(
            model="openai:gpt-4o-mini",
            user="A" * 100000,  # Very long text
            expected_output_tokens=20000
        )
        
        result = estimation_service.estimate(request)
        
        # Should have context warning (90%+ utilization)
        assert any("High context utilization" in warning for warning in result.warnings)
    
    def test_approximation_warning(self, estimation_service):
        """Test approximation warning for non-exact tokenizers."""
        request = EstimateRequest(
            model="anthropic:claude-3-5-sonnet",
            user="Test prompt",
            expected_output_tokens=100
        )
        
        result = estimation_service.estimate(request)
        
        assert result.tokenizer.approx is True
        assert any("approximation" in warning for warning in result.warnings)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz/")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "models" in data
    
    def test_estimate_endpoint(self, client):
        """Test estimate endpoint."""
        request_data = {
            "model": "openai:gpt-4o-mini",
            "system": "You are helpful.",
            "user": "Summarize the article.",
            "expected_output_tokens": 250
        }
        
        response = client.post("/estimate/", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model"] == "openai:gpt-4o-mini"
        assert data["input_tokens"] > 0
        assert data["output_tokens"] == 250
        assert data["cost"] > 0
    
    def test_batch_estimate_endpoint(self, client):
        """Test batch estimate endpoint."""
        request_data = {
            "requests": [
                {
                    "model": "openai:gpt-4o-mini",
                    "user": "First question",
                    "expected_output_tokens": 100
                },
                {
                    "model": "openai:gpt-4o-mini",
                    "user": "Second question",
                    "expected_output_tokens": 200
                }
            ]
        }
        
        response = client.post("/estimate/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_cost"] > 0
        assert data["total_input_tokens"] > 0
        assert data["total_output_tokens"] == 300
    
    def test_models_endpoint(self, client):
        """Test models endpoint."""
        response = client.get("/models/")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
    
    def test_estimate_unknown_model(self, client):
        """Test estimate with unknown model."""
        request_data = {
            "model": "unknown:model",
            "user": "Test",
            "expected_output_tokens": 100
        }
        
        response = client.post("/estimate/", json=request_data)
        assert response.status_code == 400
    
    def test_estimate_negative_tokens(self, client):
        """Test estimate with negative output tokens."""
        request_data = {
            "model": "openai:gpt-4o-mini",
            "user": "Test",
            "expected_output_tokens": -1
        }
        
        response = client.post("/estimate/", json=request_data)
        assert response.status_code == 422  # Validation error


class TestPricingLoader:
    """Test pricing loader functionality."""
    
    def test_load_pricing_from_dict(self, sample_pricing):
        """Test loading pricing from dictionary."""
        loader = PricingLoader()
        # Convert PricingConfig objects to dicts for testing
        pricing_dict = {
            model: config.model_dump() 
            for model, config in sample_pricing.items()
        }
        loader._validate_and_store(pricing_dict)
        
        assert len(loader.pricing_data) == 3
        assert "openai:gpt-4o-mini" in loader.pricing_data
        assert "anthropic:claude-3-5-sonnet" in loader.pricing_data
    
    def test_get_model_config(self, sample_pricing):
        """Test getting model configuration."""
        loader = PricingLoader()
        # Convert PricingConfig objects to dicts for testing
        pricing_dict = {
            model: config.model_dump() 
            for model, config in sample_pricing.items()
        }
        loader._validate_and_store(pricing_dict)
        
        config = loader.get_model_config("openai:gpt-4o-mini")
        assert config is not None
        assert config.vendor == "openai"
        assert config.context == 128000
        
        config = loader.get_model_config("unknown:model")
        assert config is None
    
    def test_invalid_pricing_config(self):
        """Test invalid pricing configuration."""
        loader = PricingLoader()
        
        invalid_pricing = {
            "invalid:model": {
                "vendor": "openai",
                "input_per_1k": -1.0,  # Invalid negative cost
                "tokenizer": "o200k_base",
                "kind": "chat"
            }
        }
        
        with pytest.raises(ValueError, match="Invalid pricing config"):
            loader._validate_and_store(invalid_pricing)
