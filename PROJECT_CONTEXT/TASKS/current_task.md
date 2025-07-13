# Task: Web Interface Development - Phase 1 (Foundation)

**Status**: NOT STARTED  
**Priority**: CRITICAL  
**Estimated Time**: 3-4 days  
**Approach**: Build thin API layer over existing functionality

## Overview

Create a web-based interface for the portfolio optimizer, making all the sophisticated functionality accessible through a modern web application. Focus on API-first design using FastAPI.

## Architecture Decision

**Backend**: FastAPI (already in requirements)
- Async support for real-time updates
- Automatic API documentation
- Type safety with Pydantic
- Easy integration with existing code

**Frontend**: React (separate task - Phase 2)
- This task focuses on backend API only
- Frontend will consume the API in next phase

## Phase 1 Scope (This Task)

### 1. Core API Structure

**New Directory**: `src/web/`

```
src/web/
├── __init__.py
├── app.py              # FastAPI application
├── api/
│   ├── __init__.py
│   ├── portfolios.py   # Portfolio endpoints
│   ├── optimization.py # Optimization endpoints
│   ├── analytics.py    # Analytics endpoints
│   └── data.py         # Market data endpoints
├── models/
│   ├── __init__.py
│   ├── requests.py     # Pydantic request models
│   └── responses.py    # Pydantic response models
└── services/
    ├── __init__.py
    └── portfolio_service.py  # Business logic layer
```

### 2. Essential Endpoints

**Portfolio Management**:
- `POST /api/portfolios/` - Create portfolio from CSV/JSON
- `GET /api/portfolios/{id}` - Get portfolio details
- `PUT /api/portfolios/{id}/positions` - Update positions
- `GET /api/portfolios/{id}/analytics` - Get performance metrics
- `GET /api/portfolios/{id}/exposures` - Get exposure breakdown

**Optimization**:
- `POST /api/optimize/` - Run optimization
- `GET /api/optimize/methods` - List available methods
- `GET /api/optimize/parameters` - Get optimal parameters (from our 63-day result)
- `POST /api/optimize/backtest` - Run backtest

**Market Data**:
- `GET /api/data/search` - Search for securities
- `GET /api/data/prices/{symbol}` - Get price history
- `GET /api/data/exposures` - List available exposures

### 3. Core Implementation

**File**: `src/web/app.py`

```python
"""FastAPI application for portfolio optimizer."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .api import portfolios, optimization, analytics, data
from ..data.exposure_universe import ExposureUniverse
from ..data.providers.coordinator import RawDataProviderCoordinator

logger = logging.getLogger(__name__)

# Shared resources
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    # Load exposure universe
    resources['exposure_universe'] = ExposureUniverse.from_yaml('config/exposure_universe.yaml')
    
    # Initialize data providers
    resources['data_coordinator'] = RawDataProviderCoordinator()
    
    # Load optimal parameters
    import yaml
    with open('config/optimal_parameters_portfolio_level.yaml', 'r') as f:
        resources['optimal_params'] = yaml.safe_load(f)
    
    logger.info("Application started successfully")
    yield
    # Cleanup
    logger.info("Application shutting down")

app = FastAPI(
    title="Portfolio Optimizer API",
    description="Sophisticated portfolio optimization with exposure-based analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolios.router, prefix="/api/portfolios", tags=["portfolios"])
app.include_router(optimization.router, prefix="/api/optimize", tags=["optimization"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(data.router, prefix="/api/data", tags=["market data"])

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Portfolio Optimizer API",
        "optimal_horizon": resources['optimal_params']['optimal_horizon']
    }
```

**File**: `src/web/api/portfolios.py`

```python
"""Portfolio management endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict
import pandas as pd
import io

from ..models.requests import PortfolioCreate, PositionUpdate
from ..models.responses import PortfolioResponse, ExposureResponse
from ..services.portfolio_service import PortfolioService
from ...portfolio.portfolio import Portfolio

router = APIRouter()
service = PortfolioService()

@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(portfolio: PortfolioCreate):
    """Create a new portfolio."""
    try:
        portfolio_id = service.create_portfolio(
            name=portfolio.name,
            positions=portfolio.positions
        )
        return service.get_portfolio(portfolio_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload", response_model=PortfolioResponse)
async def upload_portfolio(file: UploadFile = File(...)):
    """Upload portfolio from CSV."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert to positions
        positions = []
        for _, row in df.iterrows():
            positions.append({
                "symbol": row["symbol"],
                "shares": row["shares"],
                "cost_basis": row.get("cost_basis", None)
            })
        
        portfolio_id = service.create_portfolio(
            name=file.filename.replace('.csv', ''),
            positions=positions
        )
        return service.get_portfolio(portfolio_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{portfolio_id}/exposures", response_model=ExposureResponse)
async def get_exposures(portfolio_id: str):
    """Get portfolio exposure breakdown."""
    try:
        exposures = service.calculate_exposures(portfolio_id)
        return ExposureResponse(
            portfolio_id=portfolio_id,
            exposures=exposures,
            chart_data=service.prepare_exposure_chart_data(exposures)
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
```

**File**: `src/web/api/optimization.py`

```python
"""Optimization endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from typing import Optional

from ..models.requests import OptimizationRequest
from ..models.responses import OptimizationResponse, OptimizationStatus
from ...optimization.portfolio_optimizer import PortfolioOptimizer
from ..app import resources

router = APIRouter()

# Store running optimizations
running_optimizations = {}

@router.post("/", response_model=OptimizationResponse)
async def run_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Run portfolio optimization."""
    try:
        # Get optimal parameters
        optimal_horizon = resources['optimal_params']['optimal_horizon']
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            exposure_universe=resources['exposure_universe'],
            data_coordinator=resources['data_coordinator']
        )
        
        # Run optimization (could be async in production)
        result = optimizer.optimize(
            tickers=request.tickers,
            method=request.method,
            constraints=request.constraints,
            lookback_days=request.lookback_days or 252,
            forecast_horizon=optimal_horizon  # Use our optimal!
        )
        
        return OptimizationResponse(
            status="completed",
            optimal_weights=result['weights'],
            expected_return=result['expected_return'],
            expected_volatility=result['expected_volatility'],
            sharpe_ratio=result['sharpe_ratio'],
            diversification_ratio=result['diversification_ratio']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/parameters")
async def get_optimal_parameters():
    """Get optimal parameters from portfolio-level optimization."""
    params = resources['optimal_params']
    
    return {
        "optimal_horizon": params['optimal_horizon'],
        "portfolio_rmse": params[f'horizon_{params["optimal_horizon"]}_results']['portfolio_rmse'],
        "volatility_methods": {
            exp: details['method'] 
            for exp, details in params[f'horizon_{params["optimal_horizon"]}_results']['volatility_parameters'].items()
        },
        "correlation_method": params[f'horizon_{params["optimal_horizon"]}_results']['correlation_parameters']['method']
    }
```

### 4. Pydantic Models

**File**: `src/web/models/requests.py`

```python
"""Request models for API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PositionCreate(BaseModel):
    symbol: str
    shares: float
    cost_basis: Optional[float] = None

class PortfolioCreate(BaseModel):
    name: str
    positions: List[PositionCreate]

class OptimizationConstraints(BaseModel):
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_volatility: Optional[float] = None
    max_positions: Optional[int] = None

class OptimizationRequest(BaseModel):
    tickers: List[str]
    method: str = "max_sharpe"  # max_sharpe, min_volatility, risk_parity
    constraints: Optional[OptimizationConstraints] = None
    lookback_days: Optional[int] = 252
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
```

**File**: `src/web/models/responses.py`

```python
"""Response models for API."""

from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class PositionResponse(BaseModel):
    symbol: str
    shares: float
    cost_basis: Optional[float]
    current_value: float
    weight: float
    return_pct: float

class PortfolioResponse(BaseModel):
    id: str
    name: str
    total_value: float
    positions: List[PositionResponse]
    created_at: datetime
    updated_at: datetime

class ExposureResponse(BaseModel):
    portfolio_id: str
    exposures: Dict[str, float]
    chart_data: Dict  # For visualization

class OptimizationResponse(BaseModel):
    status: str
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    trades: Optional[List[Dict]] = None
```

### 5. Service Layer

**File**: `src/web/services/portfolio_service.py`

```python
"""Business logic for portfolio operations."""

import uuid
from typing import Dict, List
from datetime import datetime

from ...portfolio.portfolio import Portfolio
from ...portfolio.position import Position
from ..app import resources

class PortfolioService:
    """Handles portfolio business logic."""
    
    def __init__(self):
        # In production, this would use a database
        self.portfolios: Dict[str, Portfolio] = {}
    
    def create_portfolio(self, name: str, positions: List[Dict]) -> str:
        """Create a new portfolio."""
        portfolio_id = str(uuid.uuid4())
        portfolio = Portfolio(name=name)
        
        # Add positions
        for pos_data in positions:
            position = Position(
                symbol=pos_data['symbol'],
                shares=pos_data['shares'],
                cost_basis=pos_data.get('cost_basis')
            )
            portfolio.add_position(position)
        
        self.portfolios[portfolio_id] = portfolio
        return portfolio_id
    
    def get_portfolio(self, portfolio_id: str) -> Dict:
        """Get portfolio details."""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        # Get current prices
        data_provider = resources['data_coordinator']
        end_date = datetime.now()
        
        # Convert to response format
        return {
            "id": portfolio_id,
            "name": portfolio.name,
            "total_value": portfolio.get_total_value(),
            "positions": self._format_positions(portfolio),
            "created_at": datetime.now(),  # In production, track this
            "updated_at": datetime.now()
        }
    
    def calculate_exposures(self, portfolio_id: str) -> Dict[str, float]:
        """Calculate portfolio exposures."""
        # This will use the fund exposure mappings
        # For now, return placeholder
        return {
            "us_large_equity": 0.4,
            "international_equity": 0.2,
            "broad_ust": 0.3,
            "trend_following": 0.1
        }
```

### 6. Running the Application

**New File**: `scripts/run_web.py`

```python
"""Run the web application."""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### 7. Testing

**New File**: `tests/web/test_api.py`

```python
"""Test API endpoints."""

from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

def test_health_check():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_portfolio():
    """Test portfolio creation."""
    portfolio_data = {
        "name": "Test Portfolio",
        "positions": [
            {"symbol": "VOO", "shares": 100},
            {"symbol": "BND", "shares": 200}
        ]
    }
    response = client.post("/api/portfolios/", json=portfolio_data)
    assert response.status_code == 200
    assert response.json()["name"] == "Test Portfolio"

def test_optimization():
    """Test optimization endpoint."""
    opt_request = {
        "tickers": ["VOO", "BND", "GLD"],
        "method": "max_sharpe"
    }
    response = client.post("/api/optimize/", json=opt_request)
    assert response.status_code == 200
    assert "optimal_weights" in response.json()
```

## Success Criteria

- [ ] FastAPI application structure created
- [ ] Core endpoints implemented (portfolios, optimization, data)
- [ ] Pydantic models for type safety
- [ ] Service layer for business logic
- [ ] Integration with existing optimization engine
- [ ] Optimal parameters (63-day horizon) used by default
- [ ] API documentation auto-generated
- [ ] Basic tests passing
- [ ] Application runs with `python scripts/run_web.py`

## Next Steps (Phase 2)

After this API is complete:
1. Build React frontend
2. Add WebSocket support for real-time updates
3. Implement authentication
4. Add database persistence
5. Deploy to cloud

## Key Design Decisions

1. **Thin API Layer**: The API should be a thin wrapper around existing functionality
2. **No Business Logic**: All calculations use existing modules
3. **Stateless Design**: Use REST principles, state in database (later)
4. **Configuration-Driven**: Use our optimal parameters by default
5. **Error Handling**: Graceful errors with helpful messages

This creates a professional, production-ready API that showcases the portfolio optimizer's capabilities!
