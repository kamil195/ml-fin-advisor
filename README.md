# Personal Finance Advisor with Behavior Modeling

A supervised-learning personal finance advisor that classifies transactions, forecasts expenses, and generates interpretable budget recommendations with behavioral awareness.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Generate mock data for development
python -m src.data.mock_generator

# Run tests
pytest tests/unit/ -v

# Start the API server
uvicorn src.serving.app:app --reload
```

## Project Structure

See [SPEC.md](SPEC.md) §13 for the full project layout.

## Architecture

See [SPEC.md](SPEC.md) §4 for the system architecture and technology stack.
