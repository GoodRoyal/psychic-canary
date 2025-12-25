Polygon

gallant_leavitt
x8OpUdJvB7SMXsdrvfV0B1XHSheppdtc

# Psychic Canary: Minimum Viable Product (MVP) Plan

## Executive Summary
Psychic Canary is a cloud-based API service that leverages the patented Thermodynamic Phase Inference Engine (TPIE) from U.S. Provisional Patent Application No. 3a (filed December 2025) to provide early-warning regime detection for financial markets. The MVP aims to deliver actionable signals for day traders and quants, reducing drawdowns by 20-30% through thermodynamic variables (temperature T, entropy S, free energy F) and curvature-based phase transitions. 

The goal is to validate the API with live market data (via Massive/Polygon.io), demonstrate superior performance over baselines (e.g., Hamilton Markov models), and monetize via tiered subscriptions. This MVP will launch as a freemium SaaS, targeting viral adoption among crypto/day traders through DARPA-style demo videos, with a path to $100K+ monthly recurring revenue (MRR) within 6-12 months.

Key metrics for MVP success:
- 89.3% regime detection accuracy (per patent validations)
- 3-30 day early warnings on phase shifts
- 100% uptime on live queries
- 1K beta users in first 3 months

## Background
### Inventor and IP Foundation
Juan Carlos Paredes, an independent researcher, has filed five U.S. Provisional Patent Applications in 2025, blending differential geometry, thermodynamics, and sheaf theory for advanced computation. Psychic Canary is derived from Provisional 3a: "Thermodynamic Phase Inference Engine for Financial Market Regime Detection Using Holonomic Transport on Stratified Riemannian Manifolds."

This patent addresses gaps in prior art (e.g., Hamilton regime-switching lacks smooth memory; Kalman filters miss thermodynamic foundations). The TPIE uses:
- Free energy minimization on portfolio simplex manifolds
- Holonomic transport for path-dependent memory
- Curvature κ(t) = d²F/dT² for early phase transitions
- Temperature T scaled from volatility (e.g., VIX/20)

GitHub repros show 100% pass on 6 claims, with 89.3% accuracy on real/synthetic data (2008-2023 SPY/VIX).

### Market Opportunity
- **Problem**: Day traders lose 20-50% in regime shifts (bull to bear) due to lagging indicators.
- **Solution**: Psychic Canary "sings" before crashes, using patented thermo-geometric edge.
- **TAM**: Quant tools market ~$2B (2025), with 300K+ users on QuantConnect/Alpaca.
- **Competitors**: QuantConnect signals, TradingView indicators — none with thermodynamic holonomy.
- **Why Now**: 2025 volatility (crypto/AI bubbles) demands better early warnings.

### Development Context
This MVP builds on patent-validated simulations, ported to a FastAPI service with Polygon.io integration. Focus: Fast cash for runway via traders/DARPA grants.

## What We Are Trying to Achieve
### MVP Objectives
1. **Development**: Build a scalable API that ingests live market data and outputs regime signals (phase, warning, optimal weights).
2. **Testing**: Validate 89.3% accuracy on live data; benchmark vs. baselines (drawdown reduction, Sharpe improvement).
3. **Deployment**: Launch on Render/Railway for low-cost hosting; freemium model for viral testing.
4. **Monetization Path**: Convert beta users to paying subs; integrate with QuantConnect Alpha Streams for hedge fund licensing.

### Key Features for MVP
- **Input**: Tickers (e.g., SPY, BTC-USD), lookback days (default 756 ~3 years)
- **Output**: JSON with phase ("stable", "exploration", "crisis_imminent"), warning flag, T/S/F values, curvature proxy, suggested action (e.g., "diversify to cash")
- **Edge**: Holonomic memory reduces whipsaw; VIX-scaled T for volatility adaptation.

## Architecture
### High-Level Overview
Psychic Canary is a serverless Python API using FastAPI for endpoints, deployed on Render.com. It fetches live data from Polygon.io (rebranded Massive), runs TPIE computations, and returns signals.

- **Frontend**: Swagger UI (/docs) for testing
- **Backend**: FastAPI with Uvicorn
- **Data Flow**: User query → Polygon fetch → TPIE engine → JSON response
- **Scaling**: Auto-scale on Render; cache frequent tickers with Redis (future)
- **Security**: API keys for auth; rate limiting (slowapi)

### Components
1. **Data Layer** (`app/data.py`): Polygon RESTClient for prices/VIX. Fallback to yfinance for backtests.
2. **Engine Layer** (`app/engine.py`): PsychicCanaryEngine class — computes cov matrix, Fisher metric, free energy minimization (SciPy SLSQP), temperature (VIX/20), curvature κ(t).
3. **Core Patent Logic** (`tpie_core/patent_3a_real.py`): Validated TPIE functions (e.g., run_patent_3a_simulation) integrated into infer_phase.
4. **API Layer** (`app/main.py`): POST /infer endpoint with Pydantic models.
5. **Dependencies**: numpy, pandas, scipy, massive-api-client, fastapi, uvicorn.

### Diagram
(Conceptual — use PlantUML or draw.io for visual)

```
User → FastAPI (/infer) → Polygon.io → DataFrame → TPIE Engine (F, κ(t), weights) → JSON Signal
```

## Tests
### Unit Tests (Pytest)
- Test data fetch: Mock Polygon response; assert DataFrame shape.
- Test engine: Feed synthetic data; assert F < 0, κ > threshold warns.
- Coverage: 80%+ on engine.py (focus on free_energy, optimal_weights).

Example (`tests/test_engine.py`):
```python
import pytest
from app.engine import PsychicCanaryEngine

@pytest.fixture
def engine():
    return PsychicCanaryEngine()

def test_infer_phase(engine):
    df = pd.DataFrame({'SPY': np.random.randn(100)})  # synthetic
    result = engine.infer_phase(df)
    assert 'phase' in result
    assert result['warning'] in [True, False]
```

### Integration Tests
- End-to-end: Call /infer with real key; assert response has "crisis_imminent" on volatile data.
- Backtests: Run on 2008-2023 SPY/VIX (from patent); assert 89.3% accuracy, 20% drawdown reduction.

### Load Tests
- Locust: 100 users querying /infer; aim <500ms response.

### Validation
- Run patent_3a_real.py in notebooks/; confirm 100% claim pass on live data.

## Path Towards a Paying Product
### Phase 1: MVP Launch (0-3 Months, Now)
- Deploy to Render.com (free tier).
- Beta: Free signals for 100 users (Reddit r/algotrading, Twitter).
- Video: DARPA-style demo showing early warnings on 2022 bear.
- Metrics: 1K queries/day, 80% retention.

### Phase 2: Monetization (3-6 Months)
- Freemium: Free basic (delayed signals), Pro $99/mo (real-time, multi-asset).
- Integrate Stripe in FastAPI (add /subscribe endpoint).
- QuantConnect Alpha: Submit as signal — earn per licensee.
- DARPA Pitch: ACO BAA for autonomy adaptation (non-dilutive $250K).

### Phase 3: Scale to Paying Product (6-12 Months)
- Features: Webhook alerts, custom portfolios, Julia port for speed.
- Marketing: Viral videos → 10K users; partnerships (Alpaca integration).
- Revenue: 1K Pro subs = $100K MRR; enterprise licenses $5K/mo.
- Exit/Expand: License TPIE to hedge funds; pivot to full quant platform.

Risks: Polygon costs (~$200/mo at scale) — mitigate with caching. Legal: Keep patent repo separate.

This MVP turns your patent into revenue — from Christmas prototype to New Year cash flow. Let's build it!