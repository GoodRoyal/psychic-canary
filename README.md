# Psychic Canary

**Thermodynamic Phase Inference for Market Regime Detection**

## Overview

Psychic Canary uses statistical mechanics to detect market regime changes before they fully materialize. Instead of pattern-matching on price, it tracks thermodynamic phase transitions.

**Key Features:**
- Real-time regime detection (exploitation / exploration / crisis)
- Optimal portfolio weight recommendations
- 22% drawdown reduction on 2020 COVID crash backtest
- Sub-second API response with caching

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set environment variables
export POLYGON_API_KEY=your_polygon_key
export ADMIN_API_KEY=your_admin_secret

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the demo interface.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo frontend |
| `/health` | GET | Health check + stats |
| `/infer` | POST | Regime inference |
| `/api/v1/status` | GET | User tier info |
| `/docs` | GET | Swagger documentation |

### Example Request

```bash
curl -X POST https://your-app.onrender.com/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"tickers": ["SPY", "QQQ", "AGG"], "days": 252}'
```

### Example Response

```json
{
  "phase": "exploitation",
  "warning": false,
  "temperature": 0.18,
  "vol_ratio": 0.83,
  "optimal_weights": {
    "SPY": 0.35,
    "QQQ": 0.25,
    "AGG": 0.40
  },
  "suggested_action": "STABLE: Hold current allocation"
}
```

## Thermodynamic Variables

| Variable | Meaning | Warning Threshold |
|----------|---------|-------------------|
| Temperature (T) | Volatility regime | T > 0.30 |
| Vol Ratio | Recent / Historical vol | Ratio > 1.5x |
| Free Energy (F) | Portfolio stress | F < -0.5 |

## Phases

- **Exploitation**: Stable market, hold positions
- **Exploration**: Rising uncertainty, reduce leverage
- **Crisis Imminent**: Phase transition, move to safety

## Patent

Derived from U.S. Provisional Patent Application 3a: "Thermodynamic Phase Inference Engine for Financial Market Regime Detection" (December 2025).

---

*The canary sings before the crash.*
