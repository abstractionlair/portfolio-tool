"""
Return Decomposition Framework.

This module decomposes total returns into their fundamental components:
- Inflation
- Real risk-free rate
- Spread (risk premium)

This decomposition helps understand the sources of returns and is crucial
for portfolio optimization and risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np

from .fred_data import FREDDataFetcher
from .total_returns import TotalReturnFetcher
from .exposure_universe import ExposureUniverse
from .alignment_strategies import AlignmentStrategy, AlignmentStrategyFactory

logger = logging.getLogger(__name__)


class ReturnDecomposer:
    """Decomposes returns into inflation, real risk-free rate, and spread components."""
    
    def __init__(self,
                 fred_fetcher: Optional[FREDDataFetcher] = None,
                 total_return_fetcher: Optional[TotalReturnFetcher] = None,
                 alignment_strategy: Optional[AlignmentStrategy] = None):
        """Initialize the return decomposer.
        
        Args:
            fred_fetcher: FREDDataFetcher instance
            total_return_fetcher: TotalReturnFetcher instance
            alignment_strategy: Strategy for aligning FRED data with returns
        """
        self.fred_fetcher = fred_fetcher or FREDDataFetcher()
        self.total_return_fetcher = total_return_fetcher or TotalReturnFetcher()
        
        # Use forward-fill as default strategy
        self.alignment_strategy = alignment_strategy or AlignmentStrategyFactory.create('forward_fill')
        
        # Cache for decomposition results
        self._decomposition_cache = {}
    
    def decompose_returns(
        self,
        returns: pd.Series,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "monthly",
        inflation_series: str = "cpi_all",
        risk_free_maturity: str = "3m"
    ) -> pd.DataFrame:
        """Decompose returns into inflation + real risk-free rate + spread.
        
        Args:
            returns: Series of total returns to decompose
            start_date: Start date for component data
            end_date: End date for component data
            frequency: Return frequency ('daily', 'monthly', 'quarterly', 'annual')
            inflation_series: Inflation series to use ('cpi_all', 'cpi_core', 'pce')
            risk_free_maturity: Risk-free rate maturity ('3m', '1y', '10y')
            
        Returns:
            DataFrame with columns:
            - total_return: Original returns
            - inflation: Inflation component
            - real_rf_rate: Real risk-free rate component
            - spread: Risk premium (excess return over risk-free rate)
            - nominal_rf_rate: Nominal risk-free rate (for reference)
        """
        cache_key = f"{returns.name}_{start_date}_{end_date}_{frequency}_{inflation_series}_{risk_free_maturity}"
        if cache_key in self._decomposition_cache:
            logger.debug("Using cached decomposition results")
            return self._decomposition_cache[cache_key]
        
        logger.info(f"Decomposing returns for {returns.name} from {start_date.date()} to {end_date.date()}")
        
        # Step 1: Get inflation rates and align with market returns
        inflation_rates_raw = self.fred_fetcher.get_inflation_rates_for_returns(
            start_date, end_date, frequency, inflation_series
        )
        
        # Align inflation data with returns using strategy
        if not inflation_rates_raw.empty:
            inflation_rates = self.alignment_strategy.align(returns, inflation_rates_raw)
            logger.info(f"Aligned inflation data using {self.alignment_strategy.get_name()}")
        else:
            logger.warning("No inflation data available for decomposition")
            inflation_rates = pd.Series(0, index=returns.index, name='inflation')
        
        # Step 2: Get nominal risk-free rates and align with market returns
        nominal_rf_rates_raw = self.fred_fetcher.fetch_risk_free_rate(
            start_date, end_date, risk_free_maturity, frequency
        )
        
        # Align risk-free rate data using strategy
        if not nominal_rf_rates_raw.empty:
            nominal_rf_rates = self.alignment_strategy.align(returns, nominal_rf_rates_raw)
            logger.info(f"Aligned risk-free rate data using {self.alignment_strategy.get_name()}")
        else:
            logger.warning("No risk-free rate data available for decomposition")
            nominal_rf_rates = pd.Series(0.02, index=returns.index, name='nominal_rf')  # Default 2%
        
        # Step 3: Convert risk-free rates to return form based on frequency
        rf_returns = self._convert_rates_to_returns(nominal_rf_rates, frequency)
        
        # Step 4: Calculate real risk-free rate
        # Real risk-free rate = (1 + nominal_rf) / (1 + inflation) - 1
        real_rf_rates = self._calculate_real_risk_free_rates(rf_returns, inflation_rates)
        
        # Step 5: Align all series (all are now pre-aligned to returns index)
        decomposition_df = pd.DataFrame({
            'total_return': returns,
            'inflation': inflation_rates,  # Now pre-aligned
            'nominal_rf_rate': rf_returns,  # Now pre-aligned
            'real_rf_rate': real_rf_rates   # Calculated from aligned data
        })
        
        # Only drop rows where return data itself is missing
        decomposition_df = decomposition_df.dropna(subset=['total_return'])
        
        # Log data retention statistics
        initial_count = len(returns)
        final_count = len(decomposition_df)
        retention_rate = final_count / initial_count * 100
        
        logger.info(f"Data retention: {final_count}/{initial_count} ({retention_rate:.1f}%)")
        
        if retention_rate < 90:
            logger.warning(f"Low data retention rate: {retention_rate:.1f}%. "
                          "Consider adjusting alignment strategy.")
        
        if decomposition_df.empty:
            logger.error("No overlapping data for decomposition")
            return pd.DataFrame()
        
        # Step 6: Calculate spread (risk premium over inflation and risk-free rate)
        # Spread = Total Return - Inflation - Real Risk-Free Rate
        # This isolates the compensated risk premium after removing uncompensated components
        decomposition_df['spread'] = (decomposition_df['total_return'] - 
                                     decomposition_df['inflation'] - 
                                     decomposition_df['real_rf_rate'])
        
        # Step 7: Verify decomposition (approximately)
        # Total Return ≈ Inflation + Real Risk-Free Rate + Spread
        # Note: This is an approximation due to compounding effects
        decomposition_df['reconstructed'] = (
            decomposition_df['inflation'] + 
            decomposition_df['real_rf_rate'] + 
            decomposition_df['spread']
        )
        
        # Calculate decomposition error for validation
        decomposition_df['error'] = decomposition_df['total_return'] - decomposition_df['reconstructed']
        
        # Log summary statistics
        self._log_decomposition_summary(decomposition_df, returns.name or "Asset")
        
        # Cache results
        self._decomposition_cache[cache_key] = decomposition_df
        
        return decomposition_df
    
    def decompose_portfolio_returns(
        self,
        portfolio_returns: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "monthly",
        inflation_series: str = "cpi_all",
        risk_free_maturity: str = "3m"
    ) -> Dict[str, pd.DataFrame]:
        """Decompose returns for multiple assets/exposures.
        
        Args:
            portfolio_returns: DataFrame with returns for multiple assets
            start_date: Start date for component data
            end_date: End date for component data
            frequency: Return frequency
            inflation_series: Inflation series to use
            risk_free_maturity: Risk-free rate maturity
            
        Returns:
            Dict mapping asset names to decomposition DataFrames
        """
        decompositions = {}
        
        for column in portfolio_returns.columns:
            asset_returns = portfolio_returns[column]
            decomposition = self.decompose_returns(
                asset_returns,
                start_date,
                end_date,
                frequency,
                inflation_series,
                risk_free_maturity
            )
            
            if not decomposition.empty:
                decompositions[column] = decomposition
            else:
                logger.warning(f"Failed to decompose returns for {column}")
        
        return decompositions
    
    def get_decomposition_summary(
        self,
        decomposition_df: pd.DataFrame,
        annualize: bool = True
    ) -> pd.Series:
        """Generate summary statistics for a return decomposition.
        
        Args:
            decomposition_df: DataFrame from decompose_returns()
            annualize: Whether to annualize the statistics
            
        Returns:
            Series with summary statistics
        """
        if decomposition_df.empty:
            return pd.Series()
        
        # Calculate means
        means = decomposition_df[['total_return', 'inflation', 'real_rf_rate', 'spread']].mean()
        
        # Annualize if requested
        if annualize:
            # Determine frequency from index
            freq = pd.infer_freq(decomposition_df.index)
            if freq:
                if freq.startswith('M'):  # Monthly
                    annualization_factor = 12
                elif freq.startswith('D'):  # Daily
                    annualization_factor = 252
                elif freq.startswith('Q'):  # Quarterly
                    annualization_factor = 4
                else:
                    annualization_factor = 1
                
                means = means * annualization_factor
        
        # Calculate proportions
        total_return = means['total_return']
        if abs(total_return) > 0.0001:  # Avoid division by zero
            summary = pd.Series({
                'total_return': total_return,
                'inflation': means['inflation'],
                'real_rf_rate': means['real_rf_rate'],
                'spread': means['spread'],
                'inflation_pct': means['inflation'] / total_return * 100,
                'real_rf_pct': means['real_rf_rate'] / total_return * 100,
                'spread_pct': means['spread'] / total_return * 100,
                'observations': len(decomposition_df)
            })
        else:
            summary = pd.Series({
                'total_return': total_return,
                'inflation': means['inflation'],
                'real_rf_rate': means['real_rf_rate'],
                'spread': means['spread'],
                'observations': len(decomposition_df)
            })
        
        return summary
    
    def decompose_universe_returns(
        self,
        universe: ExposureUniverse,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "monthly",
        inflation_series: str = "cpi_all",
        risk_free_maturity: str = "3m"
    ) -> Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]]:
        """Decompose returns for all exposures in a universe.
        
        Args:
            universe: ExposureUniverse object
            start_date: Start date
            end_date: End date
            frequency: Return frequency
            inflation_series: Inflation series to use
            risk_free_maturity: Risk-free rate maturity
            
        Returns:
            Dict mapping exposure IDs to decomposition results
        """
        logger.info(f"Decomposing returns for {len(universe)} exposures")
        
        # First, fetch returns for all exposures
        universe_returns = self.total_return_fetcher.fetch_universe_returns(
            universe, start_date, end_date, frequency
        )
        
        # Get inflation and risk-free rate data once (shared across all assets)
        inflation_rates = self.fred_fetcher.get_inflation_rates_for_returns(
            start_date, end_date, frequency, inflation_series
        )
        
        nominal_rf_rates = self.fred_fetcher.fetch_risk_free_rate(
            start_date, end_date, risk_free_maturity, frequency
        )
        
        # Convert rates to returns
        rf_returns = self._convert_rates_to_returns(nominal_rf_rates, frequency)
        real_rf_rates = self._calculate_real_risk_free_rates(rf_returns, inflation_rates)
        
        # Decompose each exposure's returns
        decomposition_results = {}
        
        for exposure_id, return_data in universe_returns.items():
            if not return_data['success'] or return_data['returns'].empty:
                logger.warning(f"Skipping {exposure_id} - no return data available")
                continue
            
            asset_returns = return_data['returns']
            
            # Create decomposition for this exposure
            decomposition_df = pd.DataFrame({
                'total_return': asset_returns,
                'inflation': inflation_rates,
                'nominal_rf_rate': rf_returns,
                'real_rf_rate': real_rf_rates
            })
            
            # Align and clean
            decomposition_df = decomposition_df.dropna()
            
            if not decomposition_df.empty:
                # Calculate spread
                decomposition_df['spread'] = (
                    decomposition_df['total_return'] - 
                    decomposition_df['nominal_rf_rate']
                )
                
                # Get summary
                summary = self.get_decomposition_summary(decomposition_df)
                
                decomposition_results[exposure_id] = {
                    'decomposition': decomposition_df,
                    'summary': summary,
                    'implementation': return_data['implementation']
                }
            else:
                logger.warning(f"No overlapping data for {exposure_id}")
        
        logger.info(f"Successfully decomposed returns for {len(decomposition_results)} exposures")
        return decomposition_results
    
    def _convert_rates_to_returns(self, rates: pd.Series, frequency: str) -> pd.Series:
        """Convert annualized rates to period returns.
        
        Args:
            rates: Series of annualized rates (as decimals, e.g., 0.05 for 5%)
            frequency: Return frequency
            
        Returns:
            Series of period returns
        """
        if rates.empty:
            return pd.Series()
        
        # Determine periods per year
        periods_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }.get(frequency, 12)
        
        # Convert annual rate to period return
        # Period return = (1 + annual_rate)^(1/periods_per_year) - 1
        period_returns = (1 + rates) ** (1 / periods_per_year) - 1
        
        return period_returns
    
    def _calculate_real_risk_free_rates(
        self,
        nominal_rf_returns: pd.Series,
        inflation_rates: pd.Series
    ) -> pd.Series:
        """Calculate real risk-free rates from nominal rates and inflation.
        
        Args:
            nominal_rf_returns: Nominal risk-free returns
            inflation_rates: Inflation rates at same frequency
            
        Returns:
            Series of real risk-free rates
        """
        # Align series
        aligned_data = pd.DataFrame({
            'nominal': nominal_rf_returns,
            'inflation': inflation_rates
        }).dropna()
        
        if aligned_data.empty:
            return pd.Series()
        
        # Real rate = (1 + nominal) / (1 + inflation) - 1
        real_rates = (1 + aligned_data['nominal']) / (1 + aligned_data['inflation']) - 1
        
        return real_rates
    
    def _log_decomposition_summary(self, decomposition_df: pd.DataFrame, asset_name: str):
        """Log summary statistics for a decomposition."""
        summary = self.get_decomposition_summary(decomposition_df, annualize=True)
        
        if not summary.empty and 'total_return' in summary:
            logger.info(f"Decomposition summary for {asset_name}:")
            logger.info(f"  Total Return: {summary['total_return']:.2%} (annualized)")
            logger.info(f"  - Inflation: {summary['inflation']:.2%}")
            logger.info(f"  - Real Risk-Free: {summary['real_rf_rate']:.2%}")
            logger.info(f"  - Spread: {summary['spread']:.2%}")
            
            if 'spread_pct' in summary:
                logger.info(f"  Spread represents {summary['spread_pct']:.1f}% of total return")
    
    def create_decomposition_report(
        self,
        decomposition_results: Dict[str, Dict],
        output_format: str = "dataframe"
    ) -> Union[pd.DataFrame, str]:
        """Create a formatted report of decomposition results.
        
        Args:
            decomposition_results: Results from decompose_universe_returns()
            output_format: 'dataframe' or 'text'
            
        Returns:
            Formatted report
        """
        if not decomposition_results:
            return pd.DataFrame() if output_format == "dataframe" else "No results to report"
        
        # Collect summaries
        summary_data = []
        
        for exposure_id, results in decomposition_results.items():
            if 'summary' in results and not results['summary'].empty:
                summary = results['summary']
                summary_data.append({
                    'Exposure': exposure_id,
                    'Total Return': f"{summary.get('total_return', 0):.2%}",
                    'Inflation': f"{summary.get('inflation', 0):.2%}",
                    'Real RF Rate': f"{summary.get('real_rf_rate', 0):.2%}",
                    'Spread': f"{summary.get('spread', 0):.2%}",
                    'Spread %': f"{summary.get('spread_pct', 0):.1f}%" if 'spread_pct' in summary else "N/A"
                })
        
        if output_format == "dataframe":
            return pd.DataFrame(summary_data)
        else:
            # Create text report
            report_lines = [
                "Return Decomposition Report",
                "=" * 80,
                ""
            ]
            
            for item in summary_data:
                report_lines.extend([
                    f"Exposure: {item['Exposure']}",
                    f"  Total Return: {item['Total Return']} = "
                    f"Inflation ({item['Inflation']}) + "
                    f"Real RF ({item['Real RF Rate']}) + "
                    f"Spread ({item['Spread']})",
                    f"  Spread represents {item['Spread %']} of total return",
                    ""
                ])
            
            return "\n".join(report_lines)
