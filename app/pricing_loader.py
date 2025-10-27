from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import json
import httpx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PricingConfig(BaseModel):
    """Pricing configuration for a single model."""
    
    vendor: str = Field(..., description="Vendor name (e.g., 'openai', 'anthropic')")
    context: Optional[int] = Field(None, description="Context window size")
    input_per_1k: float = Field(..., description="Input cost per 1k tokens")
    output_per_1k: Optional[float] = Field(None, description="Output cost per 1k tokens")
    tokenizer: str = Field(..., description="Tokenizer name")
    kind: str = Field(..., description="Model kind (chat, embedding, etc.)")
    
    @field_validator('input_per_1k', 'output_per_1k')
    @classmethod
    def validate_positive_costs(cls, v):
        if v is not None and v < 0:
            raise ValueError('Cost must be non-negative')
        return v


class PricingLoader:
    """Loads and validates pricing configuration."""
    
    def __init__(self, pricing_url: Optional[str] = None):
        self.pricing_url = pricing_url
        self.pricing_data: Dict[str, PricingConfig] = {}
    
    async def load_pricing(self) -> Dict[str, PricingConfig]:
        """Load pricing configuration from URL or local file."""
        try:
            if self.pricing_url:
                await self._load_from_url()
            else:
                self._load_from_file()
            
            logger.info(f"Loaded pricing for {len(self.pricing_data)} models")
            return self.pricing_data
            
        except Exception as e:
            logger.error(f"Failed to load pricing: {e}")
            raise
    
    async def _load_from_url(self) -> None:
        """Load pricing from remote URL."""
        async with httpx.AsyncClient() as client:
            response = await client.get(self.pricing_url)
            response.raise_for_status()
            data = response.json()
            self._validate_and_store(data)
    
    def _load_from_file(self) -> None:
        """Load pricing from local file."""
        pricing_file = Path("app/pricing.json")
        if not pricing_file.exists():
            raise FileNotFoundError("pricing.json not found")
        
        with open(pricing_file, 'r') as f:
            data = json.load(f)
        
        self._validate_and_store(data)
    
    def _validate_and_store(self, data: Dict[str, Any]) -> None:
        """Validate and store pricing data."""
        validated_data = {}
        
        for model_name, config_data in data.items():
            try:
                validated_config = PricingConfig(**config_data)
                validated_data[model_name] = validated_config
            except Exception as e:
                logger.error(f"Invalid pricing config for {model_name}: {e}")
                raise ValueError(f"Invalid pricing config for {model_name}: {e}")
        
        self.pricing_data = validated_data
    
    def get_model_config(self, model_name: str) -> Optional[PricingConfig]:
        """Get pricing configuration for a specific model."""
        return self.pricing_data.get(model_name)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations."""
        return {
            model_name: {
                "vendor": config.vendor,
                "context": config.context,
                "input_per_1k": config.input_per_1k,
                "output_per_1k": config.output_per_1k,
                "tokenizer": config.tokenizer,
                "kind": config.kind
            }
            for model_name, config in self.pricing_data.items()
        }
