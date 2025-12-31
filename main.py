from datetime import datetime, timedelta
import argparse
import pandas as pd  # only needed if you do extra manipulations

from portfolio_tool.data_io import load_portfolio
from portfolio_tool.market_data import get_price_history, get_sector_info, get_risk_free_rate
from portfolio_tool.analytics import (
    compute_returns,
    compute_period_returns,
    compute_cumulative_index,
    compute_ytd_contribution,
    compute_monthly_portfolio_returns,
    compute_sector_weights,
    compute_risk_metrics,
    compute_correlation_matrix,
    compute_annualized_return_and_volatility,
    compute_efficient_frontier_analysis,
)
from portfolio_tool.reporting import (
    display_yearly_results,
    display_period_results,
    summarize_ytd_contributions,
    display_risk_metrics,
)
from portfolio_tool.plots import (
    plot_cumulative_returns,
    plot_return_contributions,
    plot_monthly_returns_heatmap,
    plot_sector_allocation,
    plot_correlation_matrix,
    plot_risk_return_scatter,
    plot_efficient_frontier,
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze investment portfolio performance and generate reports.'
    )
    parser.add_argument(
        'portfolio_file',
        nargs='?',
        default='new_portfolio.csv',
        help='Path to portfolio CSV file (default: new_portfolio.csv)'
    )
    args = parser.parse_args()
    
    # Load portfolio
    portfolio = load_portfolio(args.portfolio_file)
    print("Loaded Portfolio:")
    print(portfolio)

    tickers = portfolio['ticker'].tolist()
    weights = portfolio.set_index('ticker')['weight'].to_dict()

    # Benchmark ticker (example: S&P 500)
    benchmark_ticker = "SPY"

    # define date range for price history
    today = datetime.today()
    start_date = (today - timedelta(days=365 * 6)).strftime('%Y-%m-%d')  # 6 years back
    end_date = today.strftime('%Y-%m-%d')

    # fetch prices for portfolio and benchmark
    all_tickers = tickers + [benchmark_ticker]
    prices_all = get_price_history(all_tickers, start_date, end_date)

    # Split prices into portfolio and benchmark prices
    prices_portfolio = prices_all[tickers]
    prices_benchmark = prices_all[[benchmark_ticker]]

    print("\nSample of portfolio price data:")
    print(prices_portfolio.tail())   

    # compute yearly + YTD returns for portfolio
    ticker_returns, portfolio_returns = compute_returns(
        prices_portfolio,
        weights=weights,
        years_back=6
    )
    
    # compute yearly + YTD returns for benchmark
    bench_ticker_returns, _ = compute_returns(
        prices_benchmark,
        weights=None,
        years_back=6
    )

    # extract the benchmark series
    benchmark_returns = bench_ticker_returns[benchmark_ticker]

    # display yearly comparison
    display_yearly_results(ticker_returns, portfolio_returns, benchmark_returns)

    # compute period returns
    portfolio_period = compute_period_returns(prices_portfolio, weights=weights)
    benchmark_period = compute_period_returns(prices_benchmark, weights=None)

    # display period comparison
    display_period_results(portfolio_period, benchmark_period)

    # compute and display risk metrics
    # Fetch risk-free rate (10-year Treasury yield)
    risk_free_rate = get_risk_free_rate('^TNX')
    if risk_free_rate is None:
        print("\nWarning: Could not fetch risk-free rate (^TNX). Using 0% for Sharpe ratio calculation.")
        risk_free_rate = 0.0
    else:
        print(f"\nUsing risk-free rate: {risk_free_rate*100:.2f}% (10-year Treasury yield ^TNX)")
    
    risk_metrics = compute_risk_metrics(
        portfolio_prices=prices_portfolio,
        benchmark_prices=prices_benchmark,
        portfolio_weights=weights,
        periods=[1, 3, 5],
        risk_free_rate=risk_free_rate
    )
    display_risk_metrics(risk_metrics)

    # cumulative returns plot
    portfolio_cum = compute_cumulative_index(prices_portfolio, weights=weights)
    benchmark_cum = compute_cumulative_index(prices_benchmark, weights=None)

    plot_cumulative_returns(
        portfolio_cum,
        benchmark_cum,
        benchmark_label='S&P 500 (SPY)'
    )

    # YTD contribution plot
    ytd_contrib = compute_ytd_contribution(prices_portfolio, weights=weights)

    # Text summary
    summarize_ytd_contributions(ytd_contrib, top_n=3)

    plot_return_contributions(
        ytd_contrib,
        top_n=5,
        period_label='YTD',
        weights=weights
    )

    # monthly returns heatmap
    monthly_portfolio = compute_monthly_portfolio_returns(
        prices_portfolio,
        weights=weights,
        years_back=5,
    )

    plot_monthly_returns_heatmap(
        monthly_portfolio,
        title="Monthly Portfolio Returns Heatmap (Last 5 Years)"
    )

    # sector allocation pie chart
    sector_info = get_sector_info(tickers)
    sector_weights = compute_sector_weights(tickers, weights, sector_info)
    
    plot_sector_allocation(
        sector_weights,
        title="Portfolio Allocation by Sector"
    )

    # correlation matrix heatmap
    correlation_matrix = compute_correlation_matrix(
        prices_portfolio,
        years=3  # Use last 3 years for correlation calculation
    )
    
    plot_correlation_matrix(
        correlation_matrix,
        title="Portfolio Asset Correlation Matrix (Last 3 Years)"
    )

    # risk-return scatter plot
    # Fetch additional benchmark data
    benchmark_tickers = ['SPY', 'QQQ', 'AGG', 'ACWI']
    all_benchmark_tickers = [benchmark_ticker] + benchmark_tickers
    benchmark_prices_all = get_price_history(all_benchmark_tickers, start_date, end_date)
    
    # Calculate risk-return metrics for each security
    risk_return_data = {}
    
    # Portfolio
    port_ret, port_vol = compute_annualized_return_and_volatility(
        prices_portfolio, weights=weights, years=5
    )
    if port_ret is not None and port_vol is not None:
        risk_return_data['Portfolio'] = (port_vol, port_ret)
    
    # SPY (already have this)
    spy_ret, spy_vol = compute_annualized_return_and_volatility(
        prices_benchmark, weights=None, years=5
    )
    if spy_ret is not None and spy_vol is not None:
        risk_return_data['SPY (S&P 500)'] = (spy_vol, spy_ret)
    
    # QQQ, AGG, ACWI
    for ticker in benchmark_tickers:
        if ticker in benchmark_prices_all.columns:
            ticker_prices = benchmark_prices_all[[ticker]]
            ticker_ret, ticker_vol = compute_annualized_return_and_volatility(
                ticker_prices, weights=None, years=5
            )
            if ticker_ret is not None and ticker_vol is not None:
                if ticker == 'QQQ':
                    risk_return_data['QQQ (Nasdaq-100)'] = (ticker_vol, ticker_ret)
                elif ticker == 'AGG':
                    risk_return_data['AGG (Investment-Grade Bonds)'] = (ticker_vol, ticker_ret)
                elif ticker == 'ACWI':
                    risk_return_data['ACWI (All Country World Index)'] = (ticker_vol, ticker_ret)
    
    plot_risk_return_scatter(
        risk_return_data,
        title="Risk-Return Analysis (Last 5 Years)"
    )

    # Efficient frontier analysis
    # Prepare benchmark prices dict
    benchmark_prices_dict = {}
    benchmark_prices_dict['SPY'] = prices_benchmark
    for ticker in benchmark_tickers:
        if ticker in benchmark_prices_all.columns:
            benchmark_prices_dict[ticker] = benchmark_prices_all[[ticker]]
    
    # Compute efficient frontier analysis
    ef_data = compute_efficient_frontier_analysis(
        portfolio_prices=prices_portfolio,
        portfolio_weights=weights,
        benchmark_prices_dict=benchmark_prices_dict,
        years=5
    )
    
    if ef_data:
        plot_efficient_frontier(
            ef_data,
            title="Efficient Frontier Analysis (Last 5 Years)"
        )

if __name__ == "__main__":
    main()


