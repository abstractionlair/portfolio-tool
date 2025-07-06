# Exposure Universe - Final Hierarchical Structure

## Overview
The exposure universe is organized hierarchically into 5 categories containing 16 distinct exposures. TIPS are classified as Real Assets due to their inflation-hedging properties, and traditional bonds are clearly labeled as "Nominal Fixed Income."

## Category Hierarchy

### 1. Equity Beta (5 exposures)
Traditional equity market risk premiums across geographies and market caps:
- **US Large Cap Equity Beta**
- **US Small Cap Equity Beta**
- **Developed Ex-US Large Cap Equity Beta**
- **Developed Ex-US Small Cap Equity Beta**
- **Emerging Markets Equity Beta**

### 2. Factor/Style (2 exposures)
Systematic factor strategies capturing well-documented risk premia:
- **Factor/Style - Equities**: Equity-only factors (value, momentum, quality, low vol)
- **Factor/Style - Other**: Multi-asset factors including carry across bonds, currencies, commodities

### 3. Alternatives (1 exposure)
Non-traditional systematic strategies:
- **Trend Following**: Managed futures momentum strategy

### 4. Nominal Fixed Income (3 exposures)
Fixed-rate bond exposures without inflation adjustment:
- **Short-Term US Treasuries**: 1-3 year duration
- **Broad US Treasuries**: Intermediate duration
- **Dynamic Global Bonds**: Active global fixed income

### 5. Real Assets (4 exposures)
Inflation-sensitive and tangible asset exposures:
- **Real Estate**: Public REITs
- **Commodities**: Broad basket
- **Gold**: Precious metals
- **TIPS**: Treasury Inflation-Protected Securities

## Key Design Insights

### Why TIPS Are Real Assets
- Primary purpose is inflation protection
- Returns linked to CPI changes
- Behave more like real assets than nominal bonds
- Natural hedge alongside commodities and real estate

### Why "Nominal" Fixed Income
- Distinguishes from inflation-linked bonds
- Clarifies that returns are not inflation-adjusted
- More precise categorization
- Highlights the real vs. nominal distinction

### Hierarchical Benefits
1. **Clear Organization**: 5 distinct categories with logical groupings
2. **No Ambiguity**: Each exposure clearly belongs to one category
3. **Implementation Ready**: Code can reflect this hierarchy
4. **Extensible**: Easy to add new exposures to appropriate categories

## Implementation in Code

```python
class ExposureCategory(Enum):
    EQUITY_BETA = "equity_beta"
    FACTOR_STYLE = "factor_style"
    ALTERNATIVES = "alternatives"
    NOMINAL_FIXED_INCOME = "nominal_fixed_income"
    REAL_ASSETS = "real_assets"

class ExposureUniverse:
    def __init__(self):
        self.hierarchy = {
            ExposureCategory.EQUITY_BETA: [...],
            ExposureCategory.FACTOR_STYLE: [...],
            ExposureCategory.ALTERNATIVES: [...],
            ExposureCategory.NOMINAL_FIXED_INCOME: [...],
            ExposureCategory.REAL_ASSETS: [...]  # Includes TIPS
        }
```

## Summary Statistics
- **Total Exposures**: 16
- **Categories**: 5
- **Real Assets**: 4 (includes TIPS)
- **Nominal Fixed Income**: 4 (includes Cash/Risk-Free Rate)
- **Data Availability**: All exposures have identifiable sources
