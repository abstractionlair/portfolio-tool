# Exposure Universe Quick Reference

## Total Count: 16 Exposures

### By Category:
- **Equity Beta**: 5 exposures
- **Factor/Style**: 2 exposures  
- **Alternatives**: 1 exposure
- **Nominal Fixed Income**: 4 exposures
- **Real Assets**: 4 exposures

### Complete List by Category:

**Equity Beta:**
1. US Large Cap Equity Beta
2. US Small Cap Equity Beta
3. Developed Ex-US Large Cap Equity Beta
4. Developed Ex-US Small Cap Equity Beta
5. Emerging Markets Equity Beta

**Factor/Style:**
6. Factor/Style - Equities (equity-only factors)
7. Factor/Style - Other (multi-asset factors, includes carry)

**Alternatives:**
8. Trend Following

**Nominal Fixed Income:**
9. Cash/Risk-Free Rate (for leverage cost modeling)
10. Short-Term US Treasuries
11. Broad US Treasuries
12. Dynamic Global Bonds

**Real Assets:**
13. Real Estate
14. Commodities
15. Gold
16. TIPS (Treasury Inflation-Protected Securities)

### Key Implementation Mapping:
- **Trend Following**: ABYIX/AHLIX/AQMNX/ASFYX (mutual funds with 10+ year history)
- **Cash/Risk-Free Rate**: BIL/SHV/SGOV or FRED 3-month rate
- **Factor/Style - Equities**: QMNIX or MTUM/VLUE/QUAL/USMV composite
- **Factor/Style - Other**: QSPIX (includes futures yield through carry component)
- **Most exposures**: Use average of multiple ETFs for robustness
- **All exposures**: Have identifiable, accessible data sources

### Special Purpose Exposures:
- **Cash/Risk-Free Rate**: Essential for modeling leverage costs in Return Stacked products
