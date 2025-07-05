"""Convert fund replication spreadsheet to YAML format for portfolio optimizer."""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


# Exposure category mapping from the spreadsheet
EXPOSURE_CATEGORIES = {
    'bnd': 'BONDS',
    'com': 'COMMODITIES',
    'e em v': 'EM_VALUE_EQUITY',
    'e glbl': 'GLOBAL_EQUITY',
    'e int': 'INTL_EQUITY',
    'e int v': 'INTL_VALUE_EQUITY',
    'e us l': 'US_LARGE_EQUITY',
    'e us s': 'US_SMALL_EQUITY',
    'e us sv': 'US_SMALL_VALUE_EQUITY',
    'e us v': 'US_VALUE_EQUITY',
    'ls': 'LONG_SHORT',
    'mf': 'MANAGED_FUTURES',
    're': 'REAL_ESTATE'
}

# Representative funds for each category
REPRESENTATIVE_FUNDS = {
    'PFIUX': 'PIMCO Dynamic Bond Fund',  # Active multi-sector bond fund
    'PTLDX': 'PIMCO Low Duration Fund',  # Short-term bond fund
    'VFISX': 'Vanguard Short-Term Treasury Fund',
    'VAIPX': 'Vanguard Inflation-Protected Securities Fund',
    'AGG': 'iShares Core US Aggregate Bond ETF',
    'GOVT': 'iShares US Treasury Bond ETF',
    'HYLB': 'Xtrackers USD High Yield Corporate Bond ETF',
    'COMIX': 'PIMCO Commodity Real Return Strategy Fund',
    'CCRSX': 'Dreyfus Commodity Strategies Fund',
    'DFEVX': 'DFA Emerging Markets Value Fund',
    'VT': 'Vanguard Total World Stock ETF',
    'VTMGX': 'Vanguard Total International Stock Index Fund',
    'DFIVX': 'DFA International Value Fund',
    'VFIAX': 'Vanguard 500 Index Fund',
    'VSMAX': 'Vanguard Small-Cap Index Fund',
    'SFSNX': 'Schwab Fundamental US Small Company Index Fund',
    'VRVIX': 'Vanguard Russell 1000 Value Index Fund',  # Not Victory, but Vanguard!
    'QMNIX': 'AQR Market Neutral Fund',
    'ABYIX': 'Abbey Capital Futures Strategy Fund',
    'AHLIX': 'AlphaSimplex Managed Futures Strategy Fund',
    'AQMNX': 'AQR Managed Futures Strategy Fund',
    'ASFYX': 'AlphaSimplex Managed Futures Strategy Fund Y',
    'QMHIX': 'AQR Managed Futures Strategy HV Fund',
    'DBMF': 'iMGP DBi Managed Futures Strategy ETF',
    'KMLM': 'KFA Mount Lucas Managed Futures Index Strategy ETF',
    'CSDIX': 'Castle Focus Fund',
    'DFREX': 'DFA Real Estate Securities Fund',
    'VGSNX': 'Vanguard Real Estate Index Fund'
}


def parse_replication_csv(filepath: str) -> Dict[str, Any]:
    """Parse the fund replication CSV file."""
    
    # Read CSV without headers first
    df = pd.read_csv(filepath, header=None)
    
    # Extract headers
    category_headers = df.iloc[0, 6:].tolist()
    fund_headers = df.iloc[1, 6:].tolist()
    
    # Build column mapping
    column_map = {}
    for i, (cat, fund) in enumerate(zip(category_headers, fund_headers)):
        if pd.notna(cat):
            column_map[i + 6] = {
                'category': str(cat).strip(),
                'fund': str(fund).strip() if pd.notna(fund) else ''
            }
    
    # Process fund data
    funds = {}
    
    # Find the section break (empty rows)
    section_break = None
    for i in range(2, len(df)):
        if df.iloc[i, :5].isna().all():
            section_break = i
            break
    
    # Process replication section (before the break)
    for i in range(2, section_break if section_break else len(df)):
        row = df.iloc[i]
        
        # Skip empty rows
        if pd.isna(row[0]) or pd.isna(row[1]):
            continue
        
        fund_name = str(row[0]).strip()
        ticker = str(row[1]).strip()
        leverage = float(row[3]) if pd.notna(row[3]) else 1.0
        
        # Extract exposures
        exposures = {}
        total_weight = 0.0
        
        for col_idx, col_info in column_map.items():
            value = row[col_idx]
            if pd.notna(value):
                # Convert percentage strings to float
                if isinstance(value, str) and '%' in value:
                    weight = float(value.replace('%', '')) / 100
                else:
                    weight = float(value) if value else 0.0
                
                if weight > 0:
                    category = col_info['category']
                    if category in EXPOSURE_CATEGORIES:
                        exposure_type = EXPOSURE_CATEGORIES[category]
                        exposures[exposure_type] = weight
                        total_weight += weight
        
        if exposures and ticker not in funds:  # Take first occurrence if multiple
            funds[ticker] = {
                'name': fund_name,
                'leverage': leverage,
                'exposures': exposures,
                'total_weight': round(total_weight, 4),
                'category': determine_fund_category(fund_name, exposures)
            }
    
    return funds


def determine_fund_category(name: str, exposures: Dict[str, float]) -> str:
    """Determine fund category based on name and exposures."""
    name_lower = name.lower()
    
    if 'stocksplus' in name_lower:
        return 'PIMCO StocksPLUS'
    elif 'rae plus' in name_lower:
        return 'PIMCO RAE PLUS'
    elif 'return stacked' in name_lower:
        return 'Return Stacked'
    elif 'managed futures' in name_lower or 'trend' in name_lower:
        return 'Managed Futures'
    elif 'commodity' in name_lower:
        return 'Commodities'
    elif 'real estate' in name_lower:
        return 'Real Estate'
    elif 'long-short' in name_lower or 'long/short' in name_lower:
        return 'Long/Short'
    elif len(exposures) > 2:
        return 'Multi-Asset'
    else:
        return 'Other'


def create_fund_universe_yaml(funds: Dict[str, Any], output_path: str):
    """Create YAML file with fund universe."""
    
    # Structure for YAML
    yaml_data = {
        'metadata': {
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'source': 'Fund replication analysis spreadsheet',
            'methodology': 'OLS regression against factor ETFs'
        },
        'exposure_types': {
            'BONDS': 'Fixed Income',
            'COMMODITIES': 'Commodities',
            'EM_VALUE_EQUITY': 'Emerging Markets Value Equity',
            'GLOBAL_EQUITY': 'Global Equity',
            'INTL_EQUITY': 'International Developed Equity',
            'INTL_VALUE_EQUITY': 'International Value Equity',
            'US_LARGE_EQUITY': 'US Large Cap Equity',
            'US_SMALL_EQUITY': 'US Small Cap Equity',
            'US_SMALL_VALUE_EQUITY': 'US Small Cap Value Equity',
            'US_VALUE_EQUITY': 'US Large Cap Value Equity',
            'LONG_SHORT': 'Long/Short Equity',
            'MANAGED_FUTURES': 'Managed Futures/Trend Following',
            'REAL_ESTATE': 'Real Estate'
        },
        'representative_etfs': REPRESENTATIVE_FUNDS,
        'funds': {}
    }
    
    # Add fund data
    for ticker, fund_data in sorted(funds.items()):
        yaml_data['funds'][ticker] = {
            'name': fund_data['name'],
            'category': fund_data['category'],
            'exposures': fund_data['exposures'],
            'total_notional': fund_data['leverage'],
            'replication': {
                'total_weight': fund_data['total_weight'],
                'notes': f"Leverage: {fund_data['leverage']}x"
            }
        }
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created fund universe YAML at: {output_path}")
    print(f"Total funds: {len(funds)}")
    
    # Summary statistics
    categories = {}
    for fund in funds.values():
        cat = fund['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nFunds by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


def main():
    """Convert fund replication CSV to YAML format."""
    # Input and output paths
    csv_path = "InvestmentResearch_20240619 - Fund Repl.csv"
    yaml_path = "data/fund_universe.yaml"
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Parse CSV and create YAML
    funds = parse_replication_csv(csv_path)
    create_fund_universe_yaml(funds, yaml_path)
    
    # Also create a simplified version with just key funds
    key_funds = {
        ticker: data for ticker, data in funds.items()
        if ticker in ['RSSB', 'RSST', 'RSBT', 'PSTKX', 'PCFIX', 'NTSX', 'MBXIX', 'QMHIX']
    }
    
    if key_funds:
        create_fund_universe_yaml(key_funds, "data/fund_universe_core.yaml")
        print(f"\nAlso created core fund universe with {len(key_funds)} key funds")


if __name__ == "__main__":
    main()
