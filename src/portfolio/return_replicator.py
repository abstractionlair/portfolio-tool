"""Return replication validator for fund exposure assumptions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import warnings

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    LinearRegression = None
    r2_score = None

from .exposures import ExposureType, FundDefinition
from ..data.market_data import MarketDataFetcher, calculate_returns

logger = logging.getLogger(__name__)


class ReturnReplicator:
    """Validates fund exposure assumptions through return replication.
    
    Uses linear regression to test whether a fund's returns can be replicated
    using the stated exposures to underlying asset classes.
    """
    
    def __init__(self, market_data_fetcher: Optional[MarketDataFetcher] = None):
        """Initialize with market data fetcher.
        
        Args:
            market_data_fetcher: MarketDataFetcher instance for getting price data
        """
        self.data_fetcher = market_data_fetcher or MarketDataFetcher()
        
        # Default mapping of exposure types to representative ETFs
        self.default_replication_symbols = {
            ExposureType.US_LARGE_EQUITY: "SPY",
            ExposureType.US_SMALL_EQUITY: "IWM", 
            ExposureType.US_VALUE_EQUITY: "VTV",
            ExposureType.US_SMALL_VALUE_EQUITY: "VBR",
            ExposureType.INTL_EQUITY: "VXUS",
            ExposureType.INTL_VALUE_EQUITY: "VTEB",
            ExposureType.EM_VALUE_EQUITY: "VWO",
            ExposureType.GLOBAL_EQUITY: "VT",
            ExposureType.BONDS: "AGG",
            ExposureType.US_BONDS: "AGG",
            ExposureType.INTL_BONDS: "BNDX",
            ExposureType.LONG_DURATION_BONDS: "TLT",
            ExposureType.COMMODITIES: "DJP",
            ExposureType.REAL_ESTATE: "VNQ",
            ExposureType.MANAGED_FUTURES: "DBMF",
            ExposureType.TREND_FOLLOWING: "DBMF",
            ExposureType.LONG_SHORT: "QMN",
        }
    
    def validate_fund_exposures(
        self,
        fund_symbol: str,
        fund_definition: FundDefinition,
        replication_symbols: Optional[Dict[ExposureType, str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_periods: int = 252
    ) -> Dict:
        """Validate fund exposures through return replication.
        
        Args:
            fund_symbol: Symbol of fund to validate
            fund_definition: Fund definition with stated exposures
            replication_symbols: Optional mapping of exposure types to ETF symbols
            start_date: Start date for analysis (defaults to 2 years ago)
            end_date: End date for analysis (defaults to today)
            min_periods: Minimum number of observations required
            
        Returns:
            Dictionary with replication results including:
            - r_squared: R-squared of regression
            - tracking_error: Annualized tracking error
            - coefficients: Regression coefficients
            - expected_coefficients: Expected coefficients from fund definition
            - coefficient_errors: Differences between actual and expected
            - regression_stats: Additional regression statistics
            
        Raises:
            ValueError: If insufficient data or invalid inputs
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=2*365)  # 2 years
        
        # Use default replication symbols if not provided
        if replication_symbols is None:
            replication_symbols = self.default_replication_symbols
        
        logger.info(f"Validating {fund_symbol} exposures from {start_date.date()} to {end_date.date()}")
        
        # Get all required symbols
        required_symbols = [fund_symbol]
        exposure_to_symbol = {}
        
        for exposure_type, weight in fund_definition.exposures.items():
            if exposure_type in replication_symbols:
                symbol = replication_symbols[exposure_type]
                required_symbols.append(symbol)
                exposure_to_symbol[exposure_type] = symbol
            else:
                logger.warning(f"No replication symbol available for {exposure_type}")
        
        if len(exposure_to_symbol) == 0:
            raise ValueError("No replication symbols available for any exposures")
        
        # Fetch price data
        try:
            price_data = self.data_fetcher.fetch_price_history(
                required_symbols,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch price data: {e}")
        
        # Calculate returns
        returns_data = {}
        for symbol, prices in price_data.items():
            if not prices.empty:
                returns = calculate_returns(prices, period='daily')
                if len(returns) >= min_periods:
                    returns_data[symbol] = returns
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(returns)} < {min_periods}")
        
        if fund_symbol not in returns_data:
            raise ValueError(f"Insufficient return data for fund {fund_symbol}")
        
        # Align all return series
        fund_returns = returns_data[fund_symbol]
        replicating_returns = {}
        
        for exposure_type, symbol in exposure_to_symbol.items():
            if symbol in returns_data:
                replicating_returns[exposure_type] = returns_data[symbol]
        
        if len(replicating_returns) == 0:
            raise ValueError("No valid replicating return series available")
        
        # Create aligned DataFrame
        df_data = {'fund': fund_returns}
        for exposure_type, returns in replicating_returns.items():
            df_data[exposure_type.value] = returns
        
        df = pd.DataFrame(df_data).dropna()
        
        if len(df) < min_periods:
            raise ValueError(f"Insufficient aligned data: {len(df)} < {min_periods}")
        
        # Prepare regression
        y = df['fund'].values
        X_columns = [col for col in df.columns if col != 'fund']
        X = df[X_columns].values
        
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for return replication analysis. Install with: pip install scikit-learn")
        
        # Run regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)
        
        # Calculate metrics
        y_pred = reg.predict(X)
        r_squared = r2_score(y, y_pred)
        
        residuals = y - y_pred
        tracking_error = np.std(residuals) * np.sqrt(252)  # Annualized
        
        # Extract coefficients
        coefficients = {}
        for i, col in enumerate(X_columns):
            exposure_type = ExposureType(col)
            coefficients[exposure_type] = reg.coef_[i]
        
        # Compare with expected coefficients
        expected_coefficients = fund_definition.exposures.copy()
        coefficient_errors = {}
        
        for exposure_type in coefficients.keys():
            expected = expected_coefficients.get(exposure_type, 0.0)
            actual = coefficients[exposure_type]
            coefficient_errors[exposure_type] = actual - expected
        
        # Additional regression statistics
        n = len(y)
        k = len(X_columns)
        
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # Information ratio (excess return / tracking error)
        excess_returns = residuals
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        results = {
            'fund_symbol': fund_symbol,
            'analysis_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'observations': len(df)
            },
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'tracking_error': tracking_error,
            'rmse': rmse,
            'information_ratio': information_ratio,
            'intercept': reg.intercept_,
            'coefficients': {exp.value: coef for exp, coef in coefficients.items()},
            'expected_coefficients': {exp.value: coef for exp, coef in expected_coefficients.items()},
            'coefficient_errors': {exp.value: error for exp, error in coefficient_errors.items()},
            'replication_symbols': {exp.value: symbol for exp, symbol in exposure_to_symbol.items()},
            'regression_stats': {
                'mse': mse,
                'degrees_of_freedom': n - k - 1,
                'observations': n,
                'parameters': k + 1  # Including intercept
            }
        }
        
        # Log results summary
        logger.info(f"Replication results for {fund_symbol}:")
        logger.info(f"  R-squared: {r_squared:.4f}")
        logger.info(f"  Tracking Error: {tracking_error:.4f}")
        logger.info(f"  Observations: {len(df)}")
        
        return results
    
    def batch_validate_funds(
        self,
        fund_definitions: Dict[str, FundDefinition],
        replication_symbols: Optional[Dict[ExposureType, str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_r_squared: float = 0.8
    ) -> pd.DataFrame:
        """Validate multiple funds and return summary results.
        
        Args:
            fund_definitions: Dictionary mapping symbols to fund definitions
            replication_symbols: Optional mapping of exposure types to ETF symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            min_r_squared: Minimum R-squared threshold for flagging poor fits
            
        Returns:
            DataFrame with validation results for all funds
        """
        results = []
        
        for symbol, fund_def in fund_definitions.items():
            try:
                result = self.validate_fund_exposures(
                    symbol, fund_def, replication_symbols, start_date, end_date
                )
                
                summary = {
                    'symbol': symbol,
                    'name': fund_def.name,
                    'category': fund_def.category,
                    'r_squared': result['r_squared'],
                    'adj_r_squared': result['adj_r_squared'],
                    'tracking_error': result['tracking_error'],
                    'information_ratio': result['information_ratio'],
                    'observations': result['analysis_period']['observations'],
                    'intercept': result['intercept'],
                    'good_fit': result['r_squared'] >= min_r_squared
                }
                
                # Add coefficient information
                for exp_type, coef in result['coefficients'].items():
                    summary[f'coef_{exp_type}'] = coef
                
                for exp_type, error in result['coefficient_errors'].items():
                    summary[f'error_{exp_type}'] = error
                
                results.append(summary)
                
            except Exception as e:
                logger.error(f"Failed to validate {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'name': fund_def.name,
                    'category': fund_def.category,
                    'r_squared': np.nan,
                    'adj_r_squared': np.nan,
                    'tracking_error': np.nan,
                    'information_ratio': np.nan,
                    'observations': 0,
                    'intercept': np.nan,
                    'good_fit': False,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def plot_replication_results(self, validation_result: Dict) -> None:
        """Plot replication analysis results.
        
        Args:
            validation_result: Result from validate_fund_exposures
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        # This would create diagnostic plots for the replication
        # Implementation depends on having the raw return data
        # For now, just log the key metrics
        logger.info("Plotting functionality requires matplotlib and return data storage")
        logger.info(f"Key metrics for {validation_result['fund_symbol']}:")
        logger.info(f"  R² = {validation_result['r_squared']:.4f}")
        logger.info(f"  Tracking Error = {validation_result['tracking_error']:.4f}")
        logger.info(f"  Information Ratio = {validation_result['information_ratio']:.4f}")