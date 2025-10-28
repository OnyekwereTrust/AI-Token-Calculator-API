# AI Token Calculator API

A production-ready HTTP service that estimates token usage and cost for different Large Language Models (LLMs) by model type.

<img width="3024" height="1714" alt="image" src="https://github.com/user-attachments/assets/bc4e7d4d-5b54-4179-8664-f075f7776b07" />


## Features

- **Accurate Tokenization**: Uses correct tokenizers per model (tiktoken for OpenAI, approximations for others)
- **Cost Estimation**: Calculates precise costs from editable pricing tables
- **RAG Support**: Handles embeddings and vector operations
- **Batch Processing**: Estimate multiple requests at once
- **OpenAPI Documentation**: Auto-generated docs at `/docs`
- **Production Ready**: Docker support, health checks, structured logging

## Supported Models

### Exact Tokenization (tiktoken)
- OpenAI GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- OpenAI text-embedding models

### Approximate Tokenization
- Anthropic Claude models (character-based approximation)
- Meta Llama models (character-based approximation)

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and run
git clone <repository>
cd token-calculator-api
docker-compose up -d

# Test the API
curl -X POST http://localhost:8000/estimate/ \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "system": "You are helpful.",
    "user": "Summarize the article.",
    "expected_output_tokens": 250
  }'
```

### Using Poetry

```bash
# Install dependencies
poetry install

# Run the API
poetry run python -m uvicorn app.main:app --reload

# API will be available at http://localhost:8000
```

## API Endpoints

### `POST /estimate/`
Estimate token usage and cost for a single request.

**Request:**
```json
{
  "model": "openai:gpt-4o-mini",
  "system": "You are helpful.",
  "user": "Summarize the article.",
  "tools_json": "{\"tools\":[{\"name\":\"search\",\"schema\":{}}]}",
  "expected_output_tokens": 250,
  "rag": {
    "embedding_tokens": 1000,
    "num_vectors_read": 5,
    "vector_read_fee_per_1k": 0.01
  }
}
```

**Response:**
```json
{
  "model": "openai:gpt-4o-mini",
  "tokenizer": {"name": "o200k_base", "approx": false},
  "input_tokens": 123,
  "output_tokens": 250,
  "context_limit": 128000,
  "context_utilization_pct": 0.29,
  "cost": 0.00231,
  "breakdown": {
    "model_input_cost": 0.00002,
    "model_output_cost": 0.00229,
    "embedding_cost": 0.00000,
    "vector_io_cost": 0.00000
  },
  "warnings": []
}
```

### `POST /estimate/batch`
Estimate multiple requests at once.

### `GET /models/`
List all available models and their configurations.

### `POST /prices/refresh`
Refresh pricing configuration (requires API key).

### `GET /healthz/`
Health check endpoint.

## Configuration

Set these environment variables:

```bash
PORT=8000                          # API port
PRICING_URL=https://...            # Optional: remote pricing URL
ADMIN_API_KEY=change-me            # API key for admin endpoints
CORS_ORIGINS=*                     # CORS origins (comma-separated)
LOG_LEVEL=INFO                     # Logging level
```

## Pricing Configuration

Pricing is loaded from `app/pricing.json` or a remote URL. Example:

```json
{
  "openai:gpt-4o-mini": {
    "vendor": "openai",
    "context": 128000,
    "input_per_1k": 0.15,
    "output_per_1k": 0.60,
    "tokenizer": "o200k_base",
    "kind": "chat"
  }
}
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_estimate.py -v
```

### Code Quality

```bash
# Format code
poetry run black app/ tests/

# Lint code
poetry run ruff check app/ tests/
```

### Adding New Models

1. Add model configuration to `app/pricing.json`
2. Ensure tokenizer is supported in `app/tokenizers/`
3. Test with the new model

## Docker Usage

### Build and Run

```bash
# Build image
docker build -t token-calculator-api .

# Run container
docker run -p 8000:8000 \
  -e ADMIN_API_KEY=your-secret-key \
  token-calculator-api

# Using docker-compose
docker-compose up -d
```

### Health Checks

The container includes health checks:

```bash
# Check container health
docker ps

# Check API health
curl http://localhost:8000/healthz/
```

## Known Limitations

- **Anthropic Models**: Use character-based approximation (marked with `approx: true`)
- **Llama Models**: Use character-based approximation (marked with `approx: true`)
- **Pricing Updates**: Requires API restart or refresh endpoint call
- **Rate Limiting**: Simple in-memory rate limiting (60 req/min/IP)

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request
