"""
Exposure Universe Configuration and Management.

This module handles loading and managing the exposure universe configuration
that defines abstract exposures and their implementation mappings to investable instruments.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Implementation:
    """Represents one way to implement an exposure."""
    type: str  # 'etf_average', 'fund', 'composite', 'rate_series', etc.
    tickers: Optional[List[str]] = None
    ticker: Optional[str] = None
    start_date: Optional[str] = None
    alternative: Optional[str] = None
    components: Optional[List[Dict[str, Any]]] = None
    source: Optional[str] = None
    series: Optional[str] = None
    index: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None
    
    def __post_init__(self):
        """Validate implementation data."""
        if self.type == 'etf_average' and not self.tickers:
            raise ValueError("etf_average type requires tickers list")
        if self.type == 'fund' and not self.ticker:
            raise ValueError("fund type requires ticker")
        if self.type == 'composite' and not self.components:
            raise ValueError("composite type requires components list")
        if self.type == 'rate_series' and not (self.source and self.series):
            raise ValueError("rate_series type requires source and series")
    
    def get_primary_tickers(self) -> List[str]:
        """Get the primary tickers for this implementation."""
        if self.tickers:
            return self.tickers
        elif self.ticker:
            return [self.ticker]
        elif self.components:
            return [comp.get('ticker') for comp in self.components if comp.get('ticker')]
        elif self.alternative:
            return [self.alternative]
        else:
            return []


@dataclass
class Exposure:
    """Represents an abstract exposure and its implementations."""
    id: str
    name: str
    description: str
    category: str
    implementations: List[Implementation] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate exposure data."""
        if not self.implementations:
            logger.warning(f"Exposure {self.id} has no implementations")
    
    def get_preferred_implementation(self, available_tickers: Optional[List[str]] = None) -> Optional[Implementation]:
        """Get the preferred implementation based on availability and data quality."""
        if not self.implementations:
            return None
        
        # Score implementations based on preference
        scored_impls = []
        for impl in self.implementations:
            score = self._score_implementation(impl, available_tickers)
            if score > 0:
                scored_impls.append((score, impl))
        
        if not scored_impls:
            logger.warning(f"No suitable implementations found for exposure {self.id}")
            return None
        
        # Return highest scoring implementation
        scored_impls.sort(key=lambda x: x[0], reverse=True)
        return scored_impls[0][1]
    
    def _score_implementation(self, impl: Implementation, available_tickers: Optional[List[str]] = None) -> float:
        """Score an implementation based on various criteria."""
        score = 0.0
        
        # Base score by type preference
        type_scores = {
            'fund': 100,  # Prefer mutual funds for long history
            'fund_average': 95,  # Fund averages with long history
            'etf_average': 90,  # ETF averages are reliable
            'composite': 80,  # Composites are flexible
            'rate_series': 95,  # Rate series are reliable for rates
            'index': 70,  # Indices may not be directly investable
            'notes': 0   # Notes are not implementations
        }
        score += type_scores.get(impl.type, 50)
        
        # Bonus for having start_date (longer history)
        if impl.start_date:
            try:
                start_dt = datetime.strptime(impl.start_date, '%Y-%m-%d')
                years_history = (datetime.now() - start_dt).days / 365.25
                score += min(years_history * 2, 20)  # Up to 20 bonus points for 10+ years
            except ValueError:
                pass
        
        # Check ticker availability
        if available_tickers:
            tickers = impl.get_primary_tickers()
            if tickers:
                available_count = sum(1 for ticker in tickers if ticker in available_tickers)
                availability_ratio = available_count / len(tickers)
                score *= availability_ratio  # Penalty for missing tickers
        
        return score
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers referenced by this exposure."""
        all_tickers = []
        for impl in self.implementations:
            all_tickers.extend(impl.get_primary_tickers())
        return list(set(all_tickers))  # Remove duplicates


@dataclass
class ExposureUniverseConfig:
    """Configuration settings for the exposure universe."""
    categories: Dict[str, str] = field(default_factory=dict)
    min_history_years: int = 5
    preferred_history_years: int = 10


class ExposureUniverse:
    """Manages the universe of exposures and their implementations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize exposure universe from configuration."""
        self.exposures: Dict[str, Exposure] = {}
        self.config = ExposureUniverseConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExposureUniverse':
        """Create ExposureUniverse from YAML file."""
        universe = cls()
        universe.load_from_file(config_path)
        return universe
    
    def load_from_file(self, config_path: str) -> None:
        """Load exposure universe from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self._load_exposures(data.get('exposures', []))
            self._load_config(data.get('config', {}))
            
            logger.info(f"Loaded {len(self.exposures)} exposures from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def _load_exposures(self, exposures_data: List[Dict[str, Any]]) -> None:
        """Load exposure definitions from configuration data."""
        for exp_data in exposures_data:
            try:
                # Load implementations
                implementations = []
                for impl_data in exp_data.get('implementations', []):
                    impl = Implementation(**impl_data)
                    implementations.append(impl)
                
                # Create exposure
                exposure = Exposure(
                    id=exp_data['id'],
                    name=exp_data['name'],
                    description=exp_data['description'],
                    category=exp_data['category'],
                    implementations=implementations
                )
                
                self.exposures[exposure.id] = exposure
                
            except Exception as e:
                logger.error(f"Error loading exposure {exp_data.get('id', 'unknown')}: {e}")
                continue
    
    def _load_config(self, config_data: Dict[str, Any]) -> None:
        """Load configuration settings."""
        self.config.categories = config_data.get('categories', {})
        self.config.min_history_years = config_data.get('min_history_years', 5)
        self.config.preferred_history_years = config_data.get('preferred_history_years', 10)
    
    def get_exposure(self, exposure_id: str) -> Optional[Exposure]:
        """Get exposure by ID."""
        return self.exposures.get(exposure_id)
    
    def get_exposures_by_category(self, category: str) -> List[Exposure]:
        """Get all exposures in a category."""
        return [exp for exp in self.exposures.values() if exp.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories."""
        return list(set(exp.category for exp in self.exposures.values()))
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers referenced across all exposures."""
        all_tickers = []
        for exposure in self.exposures.values():
            all_tickers.extend(exposure.get_all_tickers())
        return list(set(all_tickers))  # Remove duplicates
    
    def validate_ticker_availability(self, available_tickers: List[str]) -> Dict[str, Any]:
        """Validate which exposures can be implemented with available tickers."""
        results = {
            'implementable': [],
            'partial': [],
            'missing': [],
            'summary': {}
        }
        
        for exposure in self.exposures.values():
            preferred_impl = exposure.get_preferred_implementation(available_tickers)
            if preferred_impl:
                impl_tickers = preferred_impl.get_primary_tickers()
                if all(ticker in available_tickers for ticker in impl_tickers):
                    results['implementable'].append({
                        'exposure_id': exposure.id,
                        'name': exposure.name,
                        'implementation_type': preferred_impl.type,
                        'tickers': impl_tickers
                    })
                else:
                    missing_tickers = [t for t in impl_tickers if t not in available_tickers]
                    results['partial'].append({
                        'exposure_id': exposure.id,
                        'name': exposure.name,
                        'implementation_type': preferred_impl.type,
                        'available_tickers': [t for t in impl_tickers if t in available_tickers],
                        'missing_tickers': missing_tickers
                    })
            else:
                results['missing'].append({
                    'exposure_id': exposure.id,
                    'name': exposure.name,
                    'reason': 'No suitable implementation found'
                })
        
        # Summary statistics
        results['summary'] = {
            'total_exposures': len(self.exposures),
            'fully_implementable': len(results['implementable']),
            'partially_implementable': len(results['partial']),
            'missing': len(results['missing'])
        }
        
        return results
    
    def get_implementation_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Get a matrix showing implementation details for all exposures."""
        matrix = {}
        
        for exposure in self.exposures.values():
            matrix[exposure.id] = {
                'name': exposure.name,
                'category': exposure.category,
                'implementations': []
            }
            
            for impl in exposure.implementations:
                impl_info = {
                    'type': impl.type,
                    'tickers': impl.get_primary_tickers(),
                    'start_date': impl.start_date,
                    'description': impl.description
                }
                matrix[exposure.id]['implementations'].append(impl_info)
        
        return matrix
    
    def __len__(self) -> int:
        """Return number of exposures."""
        return len(self.exposures)
    
    def __iter__(self):
        """Iterate over exposures."""
        return iter(self.exposures.values())
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ExposureUniverse({len(self.exposures)} exposures, {len(self.get_all_categories())} categories)"