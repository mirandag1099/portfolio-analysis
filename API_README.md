# Portfolio Analysis API Server

This API server wraps the Portfolio Analysis Python code and exposes it as a REST API for use with the Next.js frontend.

## Setup

1. **Install API dependencies** (in addition to your existing portfolio_tool dependencies):

```bash
pip install -r requirements_api.txt
```

Or install manually:
```bash
pip install fastapi uvicorn[standard] pydantic
```

2. **Start the API server**:

**Option 1: Using uvicorn directly (recommended for reload):**
```bash
cd "/Users/edmundo/Desktop/Projects/Portfolio Analysis"
uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
```

**Option 2: Using Python script:**
```bash
cd "/Users/edmundo/Desktop/Projects/Portfolio Analysis"
python api_server.py
```

Or if you're already in the Projects directory:
```bash
cd Portfolio\ Analysis
uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001`

- API endpoint: `http://localhost:8001/analyze`
- Health check: `http://localhost:8001/health`
- API docs: `http://localhost:8001/docs` (FastAPI auto-generated Swagger UI)

## Frontend Configuration

The Next.js frontend is configured to call `http://localhost:8001` by default. You can override this by setting the `PYTHON_API_URL` environment variable:

```bash
# In your .env.local file or environment
PYTHON_API_URL=http://localhost:8001
```

## Testing

Test the API directly:

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"portfolioText": "AAPL,25%\nGOOGL,20%\nMSFT,15%"}'
```

## Notes

- The API uses the same market data and analytics functions as `main.py`
- All portfolio calculations use real market data (via Yahoo Finance)
- Results will match exactly what you get when running `main.py` directly
- The API automatically handles portfolio CSV parsing and normalization

