#!/usr/bin/env python3
"""
Test script to verify YTD period returns match YTD contributions.

This script validates that:
1. YTD return from compute_period_returns() matches the sum of contributions from compute_ytd_contribution()
2. All non-cash assets appear in the contributors output
3. Cash contributes 0.0 but is included in the output
"""

import pandas as pd
import numpy as np
from portfolio_tool.analytics import compute_period_returns, compute_ytd_contribution, get_effective_start_date
from portfolio_tool.market_data import get_price_history
from datetime import datetime, timedelta

def test_ytd_consistency():
    """Test YTD period returns match YTD contributions."""
    
    # Mock portfolio: 3 assets + cash (4 holdings total)
    weights = {
        'AAPL': 0.25,
        'MSFT': 0.25,
        'TSLA': 0.25,
        'Cash': 0.25
    }
    
    print("=" * 80)
    print("YTD Consistency Test")
    print("=" * 80)
    print(f"\nPortfolio weights: {weights}")
    
    # Fetch price data for invested tickers (exclude cash)
    invested_tickers = [t for t in weights.keys() if t != 'Cash']
    today = datetime.today()
    start_date = (today - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\nFetching price data from {start_date} to {end_date}...")
    prices_all = get_price_history(invested_tickers, start_date, end_date)
    prices_portfolio = prices_all[invested_tickers]
    
    # Get effective start date and filter prices
    effective_start = get_effective_start_date(prices_portfolio)
    print(f"Effective start date: {effective_start}")
    prices_filtered = prices_portfolio[prices_portfolio.index >= effective_start].sort_index()
    
    print(f"Price data range: {prices_filtered.index[0]} to {prices_filtered.index[-1]}")
    print(f"Number of trading days: {len(prices_filtered)}")
    
    # Compute YTD period return
    print("\n" + "-" * 80)
    print("Computing YTD period return...")
    period_returns = compute_period_returns(prices_filtered, weights=weights)
    ytd_return = period_returns.get('YTD', None)
    
    if ytd_return is None:
        print("❌ YTD period return is None (insufficient data)")
        return False
    
    print(f"YTD Period Return: {ytd_return:.6f} ({ytd_return*100:.4f}%)")
    
    # Compute YTD contributions
    print("\n" + "-" * 80)
    print("Computing YTD contributions...")
    contributions = compute_ytd_contribution(prices_filtered, weights=weights)
    
    print("\nContributions by ticker:")
    for ticker, contrib in contributions.items():
        print(f"  {ticker:8s}: {contrib:10.6f} ({contrib*100:8.4f}%)")
    
    contribution_sum = contributions.sum()
    print(f"\nSum of contributions: {contribution_sum:.6f} ({contribution_sum*100:.4f}%)")
    
    # Verify consistency
    print("\n" + "-" * 80)
    print("Verification:")
    print(f"YTD Period Return:    {ytd_return:.10f}")
    print(f"Sum of Contributions: {contribution_sum:.10f}")
    difference = abs(ytd_return - contribution_sum)
    print(f"Difference:            {difference:.10f}")
    
    tolerance = 1e-6
    if difference <= tolerance:
        print(f"\n✅ PASS: Contributions sum matches YTD return (within {tolerance})")
    else:
        print(f"\n❌ FAIL: Contributions sum does NOT match YTD return (diff > {tolerance})")
        return False
    
    # Verify all non-cash assets appear in contributions
    print("\n" + "-" * 80)
    print("Checking non-cash assets in contributions...")
    missing_assets = []
    for ticker in invested_tickers:
        if ticker not in contributions.index:
            missing_assets.append(ticker)
            print(f"  ❌ {ticker} is missing from contributions")
        else:
            print(f"  ✅ {ticker} present in contributions")
    
    if missing_assets:
        print(f"\n❌ FAIL: Missing assets in contributions: {missing_assets}")
        return False
    
    # Verify cash contributes 0.0
    print("\n" + "-" * 80)
    print("Checking cash contribution...")
    if 'Cash' in contributions.index:
        cash_contrib = contributions['Cash']
        if abs(cash_contrib) < 1e-10:
            print(f"  ✅ Cash contribution is 0.0 (as expected)")
        else:
            print(f"  ❌ Cash contribution is {cash_contrib:.10f} (should be 0.0)")
            return False
    else:
        print(f"  ❌ Cash is missing from contributions")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_ytd_consistency()
    exit(0 if success else 1)









