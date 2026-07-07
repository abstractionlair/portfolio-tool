#!/usr/bin/env python3
"""
Quick test to verify data availability for exposure universe tickers.
"""

import yfinance as yf
from datetime import datetime, timedelta

# Key tickers to test, especially mutual funds with long history
test_tickers = {
    "Trend Following Mutual Funds": ["ABYIX", "AHLIX", "AQMNX", "ASFYX"],
    "Factor Funds": ["QMNIX", "QSPIX"],
    "PIMCO Funds": ["PFIUX", "PFUIX"],
    "Cash/Risk-Free ETFs": ["BIL", "SHV", "SGOV"],
    "Standard ETFs": ["SPY", "TLT", "VNQ"],
    "Factor ETFs": ["MTUM", "VLUE", "QUAL", "USMV"],
}

print("Testing ticker availability in yfinance...")
print("=" * 60)

# Test date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)  # 10 years

results = {}

for category, tickers in test_tickers.items():
    print(f"\n{category}:")
    print("-" * 40)
    
    for ticker in tickers:
        try:
            # Try to fetch data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                first_date = data.index[0].strftime('%Y-%m-%d')
                last_date = data.index[-1].strftime('%Y-%m-%d')
                years = (data.index[-1] - data.index[0]).days / 365.25
                
                # Check if we have adjusted close
                has_adj_close = 'Adj Close' in data.columns
                
                print(f"✓ {ticker:6} - {len(data):4} days, {years:4.1f} years "
                      f"({first_date} to {last_date}) "
                      f"{'Adj Close: Yes' if has_adj_close else 'Adj Close: NO!'}")
                
                results[ticker] = {
                    'success': True,
                    'years': years,
                    'has_adj_close': has_adj_close
                }
            else:
                print(f"✗ {ticker:6} - No data returned")
                results[ticker] = {'success': False}
                
        except Exception as e:
            print(f"✗ {ticker:6} - Error: {str(e)}")
            results[ticker] = {'success': False, 'error': str(e)}

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

successful = sum(1 for r in results.values() if r.get('success', False))
total = len(results)

print(f"Successfully retrieved: {successful}/{total} tickers")

# Check critical tickers
critical_missing = []
for ticker, result in results.items():
    if not result.get('success', False):
        if ticker in ["ABYIX", "AHLIX", "AQMNX", "ASFYX", "QMNIX", "QSPIX"]:
            critical_missing.append(ticker)

if critical_missing:
    print(f"\n⚠️  CRITICAL: Missing key mutual funds: {', '.join(critical_missing)}")
    print("We may need alternative data sources for these.")

# Check for sufficient history
insufficient_history = []
for ticker, result in results.items():
    if result.get('success', False) and result.get('years', 0) < 5:
        insufficient_history.append(f"{ticker} ({result['years']:.1f} years)")

if insufficient_history:
    print(f"\n⚠️  Insufficient history (<5 years): {', '.join(insufficient_history)}")

print("\nRecommendations:")
if critical_missing:
    print("- Need to find alternative data sources for missing mutual funds")
    print("- Consider using Tiingo, Alpha Vantage, or direct fund company APIs")
else:
    print("- All critical tickers available in yfinance ✓")
    
print("- For FRED data (risk-free rate), will need pandas_datareader")
print("- May want to implement fallback data sources for robustness")
