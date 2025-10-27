from ..pricing_loader import PricingConfig
from ..schemas import (
    CostBreakdown,
    EstimateRequest,
    EstimateResponse,
    RAGConfig,
    TokenizerInfo,
)
from ..tokenizers import tokenizer_factory


class EstimationService:
    """Service for estimating token usage and costs."""

    def __init__(self, pricing_configs: dict):
        self.pricing_configs = pricing_configs

    def estimate(self, request: EstimateRequest) -> EstimateResponse:
        """Estimate token usage and cost for a request."""
        # Get pricing configuration
        pricing_config = self.pricing_configs.get(request.model)
        if not pricing_config:
            raise ValueError(f"Unknown model: {request.model}")

        # Count input tokens
        input_tokens = self._count_input_tokens(request, pricing_config.tokenizer)

        # Calculate costs
        cost_breakdown = self._calculate_costs(
            input_tokens,
            request.expected_output_tokens,
            pricing_config,
            request.rag,
        )

        # Calculate context utilization
        context_limit = pricing_config.context
        context_utilization_pct = None
        if context_limit:
            total_tokens = input_tokens + request.expected_output_tokens
            context_utilization_pct = (total_tokens / context_limit) * 100

        # Generate warnings
        warnings = self._generate_warnings(
            input_tokens,
            request.expected_output_tokens,
            context_limit,
            pricing_config.tokenizer,
        )

        # Get tokenizer info
        _, is_approx = tokenizer_factory.count_tokens("test", pricing_config.tokenizer)
        tokenizer_info = TokenizerInfo(
            name=pricing_config.tokenizer,
            approx=is_approx,
        )

        return EstimateResponse(
            model=request.model,
            tokenizer=tokenizer_info,
            input_tokens=input_tokens,
            output_tokens=request.expected_output_tokens,
            context_limit=context_limit,
            context_utilization_pct=context_utilization_pct,
            cost=cost_breakdown.model_input_cost + cost_breakdown.model_output_cost +
                 cost_breakdown.embedding_cost + cost_breakdown.vector_io_cost,
            breakdown=cost_breakdown,
            warnings=warnings,
        )

    def _count_input_tokens(self, request: EstimateRequest, tokenizer_name: str) -> int:
        """Count total input tokens."""
        total_tokens = 0

        # Count system prompt tokens
        if request.system:
            tokens, _ = tokenizer_factory.count_tokens(request.system, tokenizer_name)
            total_tokens += tokens

        # Count user prompt tokens
        if request.user:
            tokens, _ = tokenizer_factory.count_tokens(request.user, tokenizer_name)
            total_tokens += tokens

        # Count tools JSON tokens
        if request.tools_json:
            tokens, _ = tokenizer_factory.count_tokens(request.tools_json, tokenizer_name)
            total_tokens += tokens

        return total_tokens

    def _calculate_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        pricing_config: PricingConfig,
        rag_config: RAGConfig = None,
    ) -> CostBreakdown:
        """Calculate cost breakdown."""
        # Model input cost
        model_input_cost = (input_tokens / 1000) * pricing_config.input_per_1k

        # Model output cost
        model_output_cost = 0
        if pricing_config.output_per_1k:
            model_output_cost = (output_tokens / 1000) * pricing_config.output_per_1k

        # Embedding cost
        embedding_cost = 0
        if rag_config and rag_config.embedding_tokens > 0:
            # Find embedding model pricing
            embedding_model = None
            for model_name, config in self.pricing_configs.items():
                if config.kind == "embedding" and config.vendor == pricing_config.vendor:
                    embedding_model = config
                    break

            if embedding_model:
                embedding_cost = (rag_config.embedding_tokens / 1000) * embedding_model.input_per_1k

        # Vector I/O cost
        vector_io_cost = 0
        if rag_config and rag_config.num_vectors_read > 0:
            vector_io_cost = rag_config.num_vectors_read * rag_config.vector_read_fee_per_1k

        return CostBreakdown(
            model_input_cost=model_input_cost,
            model_output_cost=model_output_cost,
            embedding_cost=embedding_cost,
            vector_io_cost=vector_io_cost,
        )

    def _generate_warnings(
        self,
        input_tokens: int,
        output_tokens: int,
        context_limit: int,
        tokenizer_name: str,
    ) -> list[str]:
        """Generate warnings for the estimation."""
        warnings = []

        # Context utilization warning
        if context_limit:
            total_tokens = input_tokens + output_tokens
            utilization_pct = (total_tokens / context_limit) * 100
            if utilization_pct >= 90:
                warnings.append(f"High context utilization: {utilization_pct:.1f}% of {context_limit:,} tokens")

        # Approximation warning
        _, is_approx = tokenizer_factory.count_tokens("test", tokenizer_name)
        if is_approx:
            warnings.append(f"Tokenizer '{tokenizer_name}' uses approximation - actual token counts may vary")

        return warnings
