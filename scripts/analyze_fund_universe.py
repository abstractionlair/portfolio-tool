"""Quick analysis of the imported fund universe."""

import yaml

# Load the fund universe
with open('data/fund_universe.yaml', 'r') as f:
    fund_data = yaml.safe_load(f)

print(f"Total funds imported: {len(fund_data['funds'])}")

# Check for key funds
key_funds = ['RSSB', 'RSST', 'RSBT', 'PSTKX', 'PCFIX', 'NTSX', 'MBXIX', 'QMHIX']
print("\nKey funds check:")
for ticker in key_funds:
    if ticker in fund_data['funds']:
        fund = fund_data['funds'][ticker]
        print(f"\n{ticker}: {fund['name']}")
        print(f"  Leverage: {fund['total_notional']}x")
        for exp_type, weight in fund['exposures'].items():
            print(f"  {exp_type}: {weight * 100:.1f}%")
    else:
        print(f"{ticker}: NOT FOUND")

# Show all funds by category
categories = {}
for ticker, fund in fund_data['funds'].items():
    cat = fund['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(ticker)

print("\n\nAll funds by category:")
for cat, tickers in sorted(categories.items()):
    print(f"\n{cat} ({len(tickers)} funds):")
    for ticker in sorted(tickers):
        print(f"  - {ticker}: {fund_data['funds'][ticker]['name']}")
