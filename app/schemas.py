from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class EstimateRequest(BaseModel):
    """Request model for token estimation."""
    
    model: str = Field(..., description="Model identifier (e.g., 'openai:gpt-4o-mini')")
    system: Optional[str] = Field(None, description="System prompt")
    user: Optional[str] = Field(None, description="User prompt")
    tools_json: Optional[str] = Field(None, description="Tools JSON string")
    expected_output_tokens: int = Field(..., description="Expected output token count")
    rag: Optional['RAGConfig'] = Field(None, description="RAG configuration")
    
    @field_validator('expected_output_tokens')
    @classmethod
    def validate_output_tokens(cls, v):
        if v < 0:
            raise ValueError('Expected output tokens must be non-negative')
        return v


class RAGConfig(BaseModel):
    """RAG configuration for embeddings and vector operations."""
    
    embedding_tokens: int = Field(0, description="Number of embedding tokens")
    num_vectors_read: int = Field(0, description="Number of vectors read")
    vector_read_fee_per_1k: float = Field(0.0, description="Vector read fee per 1k tokens")


class TokenizerInfo(BaseModel):
    """Tokenizer information."""
    
    name: str = Field(..., description="Tokenizer name")
    approx: bool = Field(..., description="Whether this is an approximation")


class CostBreakdown(BaseModel):
    """Cost breakdown by component."""
    
    model_input_cost: float = Field(..., description="Model input cost")
    model_output_cost: float = Field(..., description="Model output cost")
    embedding_cost: float = Field(..., description="Embedding cost")
    vector_io_cost: float = Field(..., description="Vector I/O cost")


class EstimateResponse(BaseModel):
    """Response model for token estimation."""
    
    model: str = Field(..., description="Model identifier")
    tokenizer: TokenizerInfo = Field(..., description="Tokenizer information")
    input_tokens: int = Field(..., description="Input token count")
    output_tokens: int = Field(..., description="Output token count")
    context_limit: Optional[int] = Field(None, description="Context window limit")
    context_utilization_pct: Optional[float] = Field(None, description="Context utilization percentage")
    cost: float = Field(..., description="Total cost")
    breakdown: CostBreakdown = Field(..., description="Cost breakdown")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


class BatchEstimateRequest(BaseModel):
    """Request model for batch estimation."""
    
    requests: List[EstimateRequest] = Field(..., description="List of estimation requests")


class BatchEstimateResponse(BaseModel):
    """Response model for batch estimation."""
    
    results: List[EstimateResponse] = Field(..., description="Individual results")
    total_cost: float = Field(..., description="Total cost across all requests")
    total_input_tokens: int = Field(..., description="Total input tokens")
    total_output_tokens: int = Field(..., description="Total output tokens")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    
    models: Dict[str, Dict[str, Any]] = Field(..., description="Available models and their configurations")


class RefreshResponse(BaseModel):
    """Response model for pricing refresh."""
    
    success: bool = Field(..., description="Whether refresh was successful")
    models_loaded: int = Field(..., description="Number of models loaded")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    ok: bool = Field(..., description="Service health status")
    models: int = Field(..., description="Number of models loaded")


# Update forward references
EstimateRequest.model_rebuild()
RAGConfig.model_rebuild()
