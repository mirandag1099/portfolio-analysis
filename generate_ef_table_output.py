#!/usr/bin/env python3
"""
Standalone script to generate a table output of efficient frontier analysis results.
This script can be run independently to see all the efficient frontier calculations
in a tabular format without modifying the main code.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_tool.data_io import load_portfolio
from portfolio_tool.market_data import get_price_history, get_risk_free_rate
from portfolio_tool.analytics import (
    compute_efficient_frontier_inputs,
    compute_efficient_frontier,
    optimize_max_sharpe,
    optimize_min_variance,
    generate_random_portfolios,
    portfolio_stats,
)

def generate_ef_table(portfolio_file='new_portfolio.csv', years=5):
    """
    Generate a comprehensive table of efficient frontier results.
    """
    print("="*80)
    print("EFFICIENT FRONTIER ANALYSIS - DETAILED TABLE OUTPUT")
    print("="*80)
    
    # Load portfolio
    portfolio = load_portfolio(portfolio_file)
    tickers = list(portfolio['ticker'])
    weights = dict(zip(portfolio['ticker'], portfolio['weight']))
    
    print(f"\nPortfolio: {portfolio_file}")
    print(f"Tickers: {tickers}")
    print(f"Weights: {weights}")
    print(f"Analysis Period: Last {years} years\n")
    
    # Get price data
    today = datetime.today()
    start_date = (today - timedelta(days=365 * (years + 2))).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    print(f"Fetching price data from {start_date} to {end_date}...")
    prices = get_price_history(tickers, start_date, end_date)
    print(f"Price records: {len(prices)}\n")
    
    # Get risk-free rate
    rf = get_risk_free_rate('^TNX')
    if rf is None:
        rf = 0.04
        print("Warning: Could not fetch risk-free rate. Using 4% as fallback.")
    else:
        print(f"Risk-free rate: {rf*100:.2f}% (10-year Treasury yield)\n")
    
    # Compute daily returns
    daily_returns = prices.pct_change().dropna()
    cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
    portfolio_daily = daily_returns[daily_returns.index >= cutoff_date]
    
    if len(portfolio_daily) < 20:
        print("ERROR: Insufficient data")
        return
    
    print(f"Analysis period: {cutoff_date.strftime('%Y-%m-%d')} to {portfolio_daily.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(portfolio_daily)}\n")
    
    # Compute inputs
    mu, cov = compute_efficient_frontier_inputs(portfolio_daily)
    
    print("="*80)
    print("1. ANNUALIZED EXPECTED RETURNS (μ)")
    print("="*80)
    mu_df = pd.DataFrame({
        'Ticker': mu.index,
        'Annualized Expected Return (%)': (mu.values * 100).round(2)
    })
    print(mu_df.to_string(index=False))
    print()
    
    print("="*80)
    print("2. ANNUALIZED COVARIANCE MATRIX (Σ)")
    print("="*80)
    cov_display = (cov * 100).round(2)  # Convert to percentage for readability
    print(cov_display.to_string())
    print()
    
    # Compute efficient frontier
    print("="*80)
    print("3. EFFICIENT FRONTIER POINTS (Sample of 20 points)")
    print("="*80)
    frontier_df = compute_efficient_frontier(mu, cov, num_points=100)
    
    # Show sample of frontier points
    sample_indices = np.linspace(0, len(frontier_df)-1, 20, dtype=int)
    frontier_sample = frontier_df.iloc[sample_indices].copy()
    frontier_sample['Return (%)'] = (frontier_sample['ret'] * 100).round(2)
    frontier_sample['Volatility (%)'] = (frontier_sample['vol'] * 100).round(2)
    frontier_sample['Sharpe Ratio'] = ((frontier_sample['ret'] - rf) / frontier_sample['vol']).round(3)
    
    display_frontier = frontier_sample[['Volatility (%)', 'Return (%)', 'Sharpe Ratio']].copy()
    print(display_frontier.to_string(index=False))
    print(f"\nTotal frontier points: {len(frontier_df)}")
    print(f"Min volatility: {frontier_df['vol'].min()*100:.2f}%")
    print(f"Max return: {frontier_df['ret'].max()*100:.2f}%")
    print()
    
    # Minimum variance portfolio
    print("="*80)
    print("4. MINIMUM VARIANCE PORTFOLIO")
    print("="*80)
    min_var_weights, min_var_ret, min_var_vol = optimize_min_variance(mu, cov)
    if min_var_weights is not None:
        min_var_df = pd.DataFrame({
            'Ticker': mu.index,
            'Weight (%)': (min_var_weights * 100).round(2)
        })
        print(min_var_df.to_string(index=False))
        print(f"\nReturn: {min_var_ret*100:.2f}%")
        print(f"Volatility: {min_var_vol*100:.2f}%")
        print(f"Sharpe Ratio: {((min_var_ret - rf) / min_var_vol):.3f}")
    print()
    
    # Tangency portfolio
    print("="*80)
    print("5. TANGENCY PORTFOLIO (Max Sharpe Ratio)")
    print("="*80)
    tangency_weights, tangency_ret, tangency_vol, tangency_sharpe = optimize_max_sharpe(mu, cov, rf)
    if tangency_weights is not None:
        tangency_df = pd.DataFrame({
            'Ticker': mu.index,
            'Weight (%)': (tangency_weights * 100).round(2)
        })
        print(tangency_df.to_string(index=False))
        print(f"\nReturn: {tangency_ret*100:.2f}%")
        print(f"Volatility: {tangency_vol*100:.2f}%")
        print(f"Sharpe Ratio: {tangency_sharpe:.3f}")
    print()
    
    # Individual assets
    print("="*80)
    print("6. INDIVIDUAL ASSETS")
    print("="*80)
    asset_data = []
    for ticker in mu.index:
        asset_ret = mu[ticker]
        asset_vol = np.sqrt(cov.loc[ticker, ticker])
        asset_sharpe = (asset_ret - rf) / asset_vol if asset_vol > 0 else None
        asset_data.append({
            'Ticker': ticker,
            'Return (%)': f"{asset_ret*100:.2f}",
            'Volatility (%)': f"{asset_vol*100:.2f}",
            'Sharpe Ratio': f"{asset_sharpe:.3f}" if asset_sharpe is not None else "N/A"
        })
    assets_df = pd.DataFrame(asset_data)
    print(assets_df.to_string(index=False))
    print()
    
    # User portfolio
    print("="*80)
    print("7. USER PORTFOLIO")
    print("="*80)
    w = pd.Series(weights).reindex(mu.index).fillna(0)
    if w.sum() > 0:
        w = w / w.sum()  # Normalize
    w_arr = w.values
    port_ret, port_vol = portfolio_stats(w_arr, mu, cov)
    port_sharpe = (port_ret - rf) / port_vol if port_vol > 0 else None
    
    portfolio_df = pd.DataFrame({
        'Ticker': mu.index,
        'Weight (%)': (w_arr * 100).round(2)
    })
    print(portfolio_df.to_string(index=False))
    print(f"\nReturn: {port_ret*100:.2f}%")
    print(f"Volatility: {port_vol*100:.2f}%")
    print(f"Sharpe Ratio: {port_sharpe:.3f}" if port_sharpe is not None else "N/A")
    print()
    
    # Random portfolios summary
    print("="*80)
    print("8. RANDOM PORTFOLIOS SUMMARY (3,000 portfolios)")
    print("="*80)
    random_portfolios = generate_random_portfolios(mu, cov, n_portfolios=3000)
    print(f"Min Return: {random_portfolios['ret'].min()*100:.2f}%")
    print(f"Max Return: {random_portfolios['ret'].max()*100:.2f}%")
    print(f"Mean Return: {random_portfolios['ret'].mean()*100:.2f}%")
    print(f"\nMin Volatility: {random_portfolios['vol'].min()*100:.2f}%")
    print(f"Max Volatility: {random_portfolios['vol'].max()*100:.2f}%")
    print(f"Mean Volatility: {random_portfolios['vol'].mean()*100:.2f}%")
    print()
    
    # Comparison table
    print("="*80)
    print("9. COMPARISON SUMMARY")
    print("="*80)
    comparison_data = []
    
    # Min variance
    if min_var_weights is not None:
        comparison_data.append({
            'Portfolio': 'Minimum Variance',
            'Return (%)': f"{min_var_ret*100:.2f}",
            'Volatility (%)': f"{min_var_vol*100:.2f}",
            'Sharpe Ratio': f"{((min_var_ret - rf) / min_var_vol):.3f}"
        })
    
    # Tangency
    if tangency_weights is not None:
        comparison_data.append({
            'Portfolio': 'Tangency (Max Sharpe)',
            'Return (%)': f"{tangency_ret*100:.2f}",
            'Volatility (%)': f"{tangency_vol*100:.2f}",
            'Sharpe Ratio': f"{tangency_sharpe:.3f}"
        })
    
    # User portfolio
    comparison_data.append({
        'Portfolio': 'Your Portfolio',
        'Return (%)': f"{port_ret*100:.2f}",
        'Volatility (%)': f"{port_vol*100:.2f}",
        'Sharpe Ratio': f"{port_sharpe:.3f}" if port_sharpe is not None else "N/A"
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    print("="*80)
    print("END OF EFFICIENT FRONTIER ANALYSIS")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate efficient frontier table output')
    parser.add_argument(
        'portfolio_file',
        nargs='?',
        default='new_portfolio.csv',
        help='Path to portfolio CSV file (default: new_portfolio.csv)'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years to analyze (default: 5)'
    )
    
    args = parser.parse_args()
    generate_ef_table(args.portfolio_file, args.years)

