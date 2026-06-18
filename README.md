# ML Fin-Advisor

A behavior-aware personal-finance machine-learning system that classifies transactions, forecasts future expenses, and generates interpretable budget recommendations through a FastAPI service and lightweight frontend.

## Project overview

ML Fin-Advisor is an end-to-end educational ML engineering project focused on personal-finance analytics. It combines transaction categorization, time-series forecasting, behavioral features, recommendation logic, model serving, testing, and deployment configuration in one repository.

The current implementation uses synthetic or demonstration data for development and portfolio purposes. It is not connected to real bank accounts and should not be treated as regulated financial advice.

## Core capabilities

- Transaction classification from merchant text and numerical features
- Expense forecasting across configurable future horizons
- Budget recommendation generation
- Behavioral and temporal feature engineering
- Confidence scores and model outputs suitable for explanation
- Consumer registration, usage history, and subscription-status endpoints
- API-key authentication and request-rate limiting
- Lemon Squeezy webhook integration for subscription lifecycle events
- Redis-backed caching with an in-memory fallback
- Docker, Docker Compose, Render, and development-container support
- Unit tests and serving checks

## Machine-learning components

The repository includes:

- TF-IDF text vectorization
- Dimensionality reduction using SVD
- Numerical feature scaling
- LightGBM-based transaction classification
- Metadata and label encoders
- Forecasting and budget-result artifacts
- Training, feature, and inference pipelines
- Classification, forecasting, recommendation, and fairness evaluation modules

Pretrained demonstration artifacts are stored in `models/serving/` so the API can be tested without retraining the entire system.

## Architecture

```text
Raw or synthetic transactions
        |
        v
Data ingestion and validation
        |
        v
Feature engineering
  - merchant text features
  - numerical features
  - temporal and behavioral features
        |
        v
ML modules
  - transaction classifier
  - expense forecast
  - budget recommender
        |
        v
FastAPI serving layer
        |
        +--> lightweight frontend
        +--> Redis or in-memory cache
        +--> subscription and webhook services
```

A more detailed technical specification is available in [`SPEC.md`](SPEC.md).

## Repository structure

```text
configs/                 Model, feature, fairness, and serving settings
frontend/                Lightweight browser-based interface
infrastructure/docker/   Serving-container configuration
models/serving/          Demonstration model artifacts
pipelines/               Training, feature, and inference pipelines
src/data/                Schemas, ingestion, and synthetic-data generation
src/evaluation/          Metrics, comparisons, and fairness auditing
src/features/            Numerical, temporal, and behavioral features
src/serving/             FastAPI application, middleware, and routes
tests/                   Unit and integration-oriented tests
Dockerfile               Main container definition
docker-compose.yml       API and Redis services
render.yaml              Render deployment configuration
SPEC.md                  Full system specification
```

## API endpoints

The service includes endpoints for:

- `POST /classify` — classify a transaction
- `GET /forecast/{user_id}` — retrieve an expense forecast
- `GET /budget/{user_id}` — retrieve budget recommendations
- `POST /consumer/register` — register a consumer user
- `POST /consumer/analyse` — analyse consumer transactions
- `GET /consumer/history/{user_id}` — retrieve analysis history
- `GET /health` — service health check
- `GET /ready` — model-readiness check
- `POST /webhooks/lemonsqueezy` — process subscription events

Interactive API documentation is available at `/docs` while the service is running.

## Local setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Replace placeholder values with your own local credentials. Never commit the real `.env` file.

### 3. Generate demonstration data

```bash
python -m src.data.mock_generator
```

### 4. Run tests

```bash
pytest tests/unit/ -v
```

### 5. Start the API

```bash
uvicorn src.serving.app:app --reload
```

Then open:

```text
http://localhost:8000/docs
```

## Docker setup

```bash
docker compose up --build
```

This starts the API and Redis services. The API is exposed on port `8000`.

## Security notes

- Real API keys, payment secrets, passwords, and `.env` files must never be committed.
- Authentication should be enabled in deployment by setting `API_KEYS`.
- Lemon Squeezy webhook verification should always use a configured webhook secret in production.
- CORS, public administrative routes, Redis exposure, and rate limits should be tightened before any real deployment.
- Model artifacts should only be loaded from trusted sources because serialized Python model files can execute code during loading.

## Current status

The repository demonstrates a broad ML engineering workflow and includes serving, subscription, deployment, and frontend components. It remains a portfolio and development project rather than a production financial product. Before real-world use, it would require persistent user storage, stronger authentication, encrypted secrets management, audited financial logic, monitoring, privacy controls, and compliance review.

## Limitations

- Uses synthetic or demonstration data
- No live bank-data integration
- Subscription and consumer stores are not yet designed as durable production databases
- Forecast and recommendation quality has not been validated on a large real-world user dataset
- Outputs are educational and should not be interpreted as professional financial advice

## Future improvements

- Add secure authentication and persistent user accounts
- Add PostgreSQL-backed storage
- Add encrypted transaction ingestion
- Add model monitoring and drift detection
- Add explainability dashboards
- Add rolling forecast evaluation
- Add CI/CD and automated security scanning
- Validate performance on consented real-world data

## Author

**Muhammad Kamil Shah**  
BS Data Science

## License

See the repository license file for usage terms.
