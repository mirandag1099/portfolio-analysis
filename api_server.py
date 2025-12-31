"""FastAPI server wrapper for Portfolio Analysis backend."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling NaN and None.
    Returns default (0.0) if value is NaN, None, or cannot be converted.
    """
    if value is None:
        return default
    if pd.isna(value):
        return default
    try:
        result = float(value)
        if pd.isna(result) or np.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

from portfolio_tool.data_io import load_portfolio
from portfolio_tool.market_data import get_price_history, get_sector_info, get_risk_free_rate
from portfolio_tool.analytics import (
    compute_returns,
    compute_period_returns,
    compute_cumulative_index,
    compute_monthly_portfolio_returns,
    compute_risk_metrics,
    compute_correlation_matrix,
    compute_annualized_return_and_volatility,
    compute_efficient_frontier_analysis,
    compute_daily_returns,
    compute_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_ulcer_index,
    compute_drawdown_series,
    compute_rolling_sharpe_ratio,
    compute_ytd_contribution,
    compute_rolling_volatility,
    compute_rolling_beta,
    get_effective_start_date,
    get_as_of_date,
    compute_ytd_risk_contribution,
    MIN_OBS_CALMAR,
)

app = FastAPI(
    title="Portfolio Analysis API",
    description="Backend API for portfolio analytics using real market data",
    version="1.0.0"
)

# CORS middleware - allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",  # Vite default port
        "http://localhost:5174",  # Vite alternate port
        "http://localhost:8080",  # Current frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortfolioRequest(BaseModel):
    portfolioText: str


def parse_portfolio_text(portfolio_text: str) -> pd.DataFrame:
    """Parse portfolio text (CSV format) into DataFrame matching load_portfolio format."""
    lines = portfolio_text.strip().split('\n')
    holdings = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and header row
        if not line or line.lower().startswith('ticker'):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            ticker = parts[0].upper()
            weight_str = parts[1].replace('%', '').strip()
            try:
                weight = float(weight_str)
                # Convert to decimal if > 1 (assume percentage)
                weight_decimal = weight / 100.0 if weight > 1.0 else weight
                holdings.append({'ticker': ticker, 'weight': weight_decimal})
            except ValueError:
                continue
    
    if not holdings:
        raise ValueError("No valid portfolio entries found")
    
    df = pd.DataFrame(holdings)
    
    # Normalize weights to sum to 1
    total_weight = df['weight'].sum()
    if total_weight > 0:
        df['weight'] = df['weight'] / total_weight
    
    return df


@app.post("/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio and return formatted results matching frontend expectations."""
    try:
        # Parse portfolio from text
        portfolio_df = parse_portfolio_text(request.portfolioText)
        
        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df.set_index('ticker')['weight'].to_dict()
        
        # Filter out cash tickers before fetching prices (cash has no price data)
        from portfolio_tool.analytics import is_cash_ticker
        invested_tickers = [ticker for ticker in tickers if not is_cash_ticker(ticker)]
        
        benchmark_ticker = "SPY"
        today = datetime.today()
        start_date = (today - timedelta(days=365 * 6)).strftime('%Y-%m-%d')  # 6 years back
        # Use tomorrow as end_date to ensure we get data through today (yfinance end parameter is exclusive)
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch real market data (only for invested tickers, not cash)
        # If no invested tickers, we still need benchmark data
        if invested_tickers:
            all_tickers = invested_tickers + [benchmark_ticker]
            prices_all = get_price_history(all_tickers, start_date, end_date)
            # Create prices_portfolio with only invested tickers (cash will be handled separately in analytics)
            prices_portfolio = prices_all[invested_tickers]
            prices_benchmark = prices_all[[benchmark_ticker]]
        else:
            # Portfolio is 100% cash - only fetch benchmark
            prices_all = get_price_history([benchmark_ticker], start_date, end_date)
            prices_portfolio = pd.DataFrame()  # Empty DataFrame for 100% cash portfolio
            prices_benchmark = prices_all[[benchmark_ticker]]
        
        # Get the actual last date from the data (accounts for weekends, holidays, and yfinance delays)
        # This ensures the displayed end date matches the actual last date in the dataset
        if not prices_portfolio.empty:
            actual_end_date = get_as_of_date(prices_portfolio.index)
        elif not prices_benchmark.empty:
            actual_end_date = get_as_of_date(prices_benchmark.index)
        else:
            actual_end_date = pd.Timestamp(today)
        
        # Determine effective start date: latest first available date across all securities
        # This makes the "latest common start date" behavior explicit rather than relying
        # on implicit dropna() behavior in return calculations
        if not prices_portfolio.empty:
            # Combine portfolio and benchmark prices to find effective start date
            # (benchmark should be included to ensure alignment)
            combined_prices = pd.concat([prices_portfolio, prices_benchmark], axis=1)
            effective_start_date = get_effective_start_date(combined_prices, default_months_back=60)
        elif not prices_benchmark.empty:
            # Portfolio is 100% cash, only use benchmark for effective start date
            effective_start_date = get_effective_start_date(prices_benchmark, default_months_back=60)
        else:
            effective_start_date = pd.Timestamp(start_date)
        
        # Filter prices to effective_start_date:end_date for consistent calculations
        # This ensures all metrics use the same date range
        if not prices_portfolio.empty and effective_start_date is not None:
            prices_portfolio = prices_portfolio[prices_portfolio.index >= effective_start_date]
        if not prices_benchmark.empty and effective_start_date is not None:
            prices_benchmark = prices_benchmark[prices_benchmark.index >= effective_start_date]
        
        # Get risk-free rate
        risk_free_rate = get_risk_free_rate('^TNX')
        if risk_free_rate is None:
            risk_free_rate = 0.02  # Default 2%
        
        # Compute yearly returns for returns table
        ticker_returns, portfolio_returns = compute_returns(
            prices_portfolio, weights=weights, years_back=6
        )
        bench_ticker_returns, _ = compute_returns(
            prices_benchmark, weights=None, years_back=6
        )
        benchmark_returns = bench_ticker_returns[benchmark_ticker]
        
        # Build returns table (yearly + YTD)
        returns_table = []
        # Use actual_end_date.year for consistency (matches the actual data range)
        current_year = actual_end_date.year if isinstance(actual_end_date, pd.Timestamp) else pd.Timestamp(actual_end_date).year
        
        # Get years from portfolio_returns index (excluding YTD)
        years = [idx for idx in portfolio_returns.index if isinstance(idx, int)]
        years.sort()
        # Include YTD if present
        if "YTD" in portfolio_returns.index:
            years.append("YTD")
        
        for period in years:
            port_ret = safe_float(portfolio_returns[period] if period in portfolio_returns.index else 0.0)
            bench_ret = safe_float(benchmark_returns[period] if period in benchmark_returns.index else 0.0)
            returns_table.append({
                "period": str(period),
                "portfolioReturn": port_ret,
                "benchmarkReturn": bench_ret
            })
        
        # Compute period returns (1M, 3M, YTD, 1Y, 3Y, 5Y)
        portfolio_period = compute_period_returns(prices_portfolio, weights=weights)
        benchmark_period = compute_period_returns(prices_benchmark, weights=None)
        
        # Map period returns - preserve None values for periods with insufficient history
        # None values indicate that the portfolio doesn't have enough history for that period
        def safe_float_or_none(value):
            """Convert to float if valid, preserve None if value is None or NaN."""
            if value is None:
                return None
            result = safe_float(value, default=None)
            # If safe_float returns default (None) for NaN/Invalid, preserve None
            return result if result is not None else None
        
        period_returns_portfolio = {
            '1M': safe_float_or_none(portfolio_period.get('1M')),
            '3M': safe_float_or_none(portfolio_period.get('3M')),
            'YTD': safe_float_or_none(portfolio_period.get('YTD')),
            '1Y': safe_float_or_none(portfolio_period.get('1Y')),
            '3Y': safe_float_or_none(portfolio_period.get('3Y')),
            '5Y': safe_float_or_none(portfolio_period.get('5Y')),
        }
        
        period_returns_benchmark = {
            '1M': safe_float_or_none(benchmark_period.get('1M')),
            '3M': safe_float_or_none(benchmark_period.get('3M')),
            'YTD': safe_float_or_none(benchmark_period.get('YTD')),
            '1Y': safe_float_or_none(benchmark_period.get('1Y')),
            '3Y': safe_float_or_none(benchmark_period.get('3Y')),
            '5Y': safe_float_or_none(benchmark_period.get('5Y')),
        }
        
        # Compute risk metrics (using 3Y period for volatility, max drawdown, and ulcer index)
        risk_metrics_df = compute_risk_metrics(
            portfolio_prices=prices_portfolio,
            benchmark_prices=prices_benchmark,
            portfolio_weights=weights,
            periods=[1, 3, 5],
            risk_free_rate=risk_free_rate
        )
        
        # Extract 3Y risk metrics for volatility, max drawdown, beta, sharpe, sortino (always use 3Y)
        # Initialize ratios as None - they will be set to None if insufficient data
        risk_metrics = {
            'annualVolatility': 0.0,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'beta': 1.0,
            'maxDrawdown': 0.0,
        }
        
        # Always use 3Y for volatility, max drawdown, sharpe, sortino, and beta (for Risk section)
        if '3Y' in risk_metrics_df.columns:
            if 'Volatility' in risk_metrics_df.index:
                vol_value = risk_metrics_df.loc['Volatility', '3Y']
                risk_metrics['annualVolatility'] = safe_float(vol_value)
            if 'Max Drawdown' in risk_metrics_df.index:
                max_dd_value = risk_metrics_df.loc['Max Drawdown', '3Y']
                risk_metrics['maxDrawdown'] = safe_float(max_dd_value)
            if 'Sharpe Ratio' in risk_metrics_df.index:
                sharpe_value = risk_metrics_df.loc['Sharpe Ratio', '3Y']
                risk_metrics['sharpeRatio'] = safe_float_or_none(sharpe_value)
            if 'Sortino Ratio' in risk_metrics_df.index:
                sortino_value = risk_metrics_df.loc['Sortino Ratio', '3Y']
                risk_metrics['sortinoRatio'] = safe_float_or_none(sortino_value)
            if 'Beta' in risk_metrics_df.index:
                beta_value = risk_metrics_df.loc['Beta', '3Y']
                risk_metrics['beta'] = safe_float(beta_value, default=1.0)
        
        # Calculate Calmar Ratio: CAGR (3Y) / |Max Drawdown (3Y)|
        # Use 3Y CAGR from period returns and 3Y max drawdown
        # Require sufficient data (at least 1 year of trading days) for reliable Calmar ratio
        calmar_ratio = None
        if period_returns_portfolio.get('3Y') is not None:
            # Compute portfolio daily returns for data sufficiency check
            portfolio_daily_for_calmar = compute_daily_returns(prices_portfolio, weights=weights)
            # Check if we have enough daily return observations in the 3Y period (1 year ≈ 252 trading days minimum)
            cutoff_date_3y = portfolio_daily_for_calmar.index[-1] - pd.DateOffset(years=3)
            period_daily_returns = portfolio_daily_for_calmar[portfolio_daily_for_calmar.index >= cutoff_date_3y]
            if len(period_daily_returns) >= MIN_OBS_CALMAR:
                cagr_3y = period_returns_portfolio['3Y']
                max_dd_abs = abs(risk_metrics['maxDrawdown']) if risk_metrics['maxDrawdown'] is not None else 0
                if max_dd_abs > 0 and not pd.isna(max_dd_abs):
                    calmar_ratio = safe_float(cagr_3y / max_dd_abs)
        risk_metrics['calmarRatio'] = calmar_ratio
        
        # Compute benchmark risk metrics (using 3Y period to match portfolio metrics)
        benchmark_daily = compute_daily_returns(prices_benchmark, weights=None)
        benchmark_cum = compute_cumulative_index(prices_benchmark, weights=None)
        
        # Benchmark risk metrics (3Y period)
        # Initialize ratios as None - they will be set to None if insufficient data
        benchmark_risk_metrics = {
            'annualVolatility': 0.0,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'maxDrawdown': 0.0,
            'beta': 1.0,  # Benchmark beta is always 1.0 (beta against itself)
        }
        
        # Benchmark volatility (3Y)
        bench_vol = compute_volatility(benchmark_daily, years=3)
        benchmark_risk_metrics['annualVolatility'] = safe_float(bench_vol)
        
        # Benchmark Sharpe ratio (3Y)
        bench_sharpe = compute_sharpe_ratio(benchmark_daily, years=3, risk_free_rate=risk_free_rate)
        benchmark_risk_metrics['sharpeRatio'] = safe_float_or_none(bench_sharpe)
        
        # Benchmark Sortino ratio (3Y)
        bench_sortino = compute_sortino_ratio(benchmark_daily, years=3, risk_free_rate=risk_free_rate)
        benchmark_risk_metrics['sortinoRatio'] = safe_float_or_none(bench_sortino)
        
        # Benchmark max drawdown (3Y)
        bench_max_dd = compute_max_drawdown(benchmark_cum, years=3)
        benchmark_risk_metrics['maxDrawdown'] = safe_float(bench_max_dd)
        
        # Benchmark beta is always 1.0 (beta of benchmark vs itself)
        benchmark_risk_metrics['beta'] = 1.0
        
        # Benchmark Calmar Ratio: CAGR (3Y) / |Max Drawdown (3Y)|
        # Require sufficient data (at least 1 year of trading days) for reliable Calmar ratio
        bench_calmar_ratio = None
        if period_returns_benchmark.get('3Y') is not None:
            # Check if we have enough daily return observations in the 3Y period (1 year ≈ 252 trading days minimum)
            cutoff_date_3y = benchmark_daily.index[-1] - pd.DateOffset(years=3)
            period_bench_daily_returns = benchmark_daily[benchmark_daily.index >= cutoff_date_3y]
            if len(period_bench_daily_returns) >= MIN_OBS_CALMAR:
                bench_cagr_3y = period_returns_benchmark['3Y']
                bench_max_dd_abs = abs(benchmark_risk_metrics['maxDrawdown']) if benchmark_risk_metrics['maxDrawdown'] is not None else 0
                if bench_max_dd_abs > 0 and not pd.isna(bench_max_dd_abs):
                    bench_calmar_ratio = safe_float(bench_cagr_3y / bench_max_dd_abs)
        benchmark_risk_metrics['calmarRatio'] = bench_calmar_ratio
        
        # Cumulative returns (growth of $1,000)
        portfolio_cum = compute_cumulative_index(prices_portfolio, weights=weights)
        benchmark_cum = compute_cumulative_index(prices_benchmark, weights=None)
        
        # Calculate 5Y cumulative return (from 60 months ago to actual_end_date)
        # Use 60 months to match exactly with the CAGR calculation period
        # Use actual_end_date for consistency (accounts for weekends, holidays, yfinance delays)
        cutoff_date_5y = actual_end_date - pd.DateOffset(months=60)
        portfolio_cum_5y = portfolio_cum[portfolio_cum.index >= cutoff_date_5y]
        benchmark_cum_5y = benchmark_cum[benchmark_cum.index >= cutoff_date_5y]
        
        cumulative_return_5y_portfolio = 0.0
        cumulative_return_5y_benchmark = 0.0
        
        if len(portfolio_cum_5y) > 0 and len(benchmark_cum_5y) > 0:
            # Get starting and ending values for 5Y period
            portfolio_start_5y = portfolio_cum_5y.iloc[0]
            portfolio_end_5y = portfolio_cum_5y.iloc[-1]
            benchmark_start_5y = benchmark_cum_5y.iloc[0]
            benchmark_end_5y = benchmark_cum_5y.iloc[-1]
            
            # Calculate cumulative return: (end / start - 1) * 100
            if portfolio_start_5y > 0:
                cumulative_return_5y_portfolio = safe_float(((portfolio_end_5y / portfolio_start_5y) - 1) * 100)
            if benchmark_start_5y > 0:
                cumulative_return_5y_benchmark = safe_float(((benchmark_end_5y / benchmark_start_5y) - 1) * 100)
        
        # Extract 5Y metrics for Performance Metrics table (after cumulative indices are computed)
        performance_metrics_5y = {
            'cumulativeReturn5Y': cumulative_return_5y_portfolio,
            'cumulativeReturn5YBenchmark': cumulative_return_5y_benchmark,
            'cagr5Y': period_returns_portfolio.get('5Y', 0.0),
            'cagr5YBenchmark': period_returns_benchmark.get('5Y', 0.0),
            'maxDrawdown5Y': 0.0,
            'maxDrawdown5YBenchmark': 0.0,
            'sharpeRatio5Y': 0.0,
            'sharpeRatio5YBenchmark': 0.0,
        }
        
        # Extract 5Y max drawdown and Sharpe ratio from risk_metrics_df
        if '5Y' in risk_metrics_df.columns:
            if 'Max Drawdown' in risk_metrics_df.index:
                max_dd_5y = risk_metrics_df.loc['Max Drawdown', '5Y']
                performance_metrics_5y['maxDrawdown5Y'] = safe_float(max_dd_5y)
            if 'Sharpe Ratio' in risk_metrics_df.index:
                sharpe_5y = risk_metrics_df.loc['Sharpe Ratio', '5Y']
                performance_metrics_5y['sharpeRatio5Y'] = safe_float(sharpe_5y)
        
        # Calculate benchmark 5Y max drawdown and Sharpe ratio
        bench_max_dd_5y = compute_max_drawdown(benchmark_cum, years=5)
        performance_metrics_5y['maxDrawdown5YBenchmark'] = safe_float(bench_max_dd_5y)
        
        bench_sharpe_5y = compute_sharpe_ratio(benchmark_daily, years=5, risk_free_rate=risk_free_rate)
        performance_metrics_5y['sharpeRatio5YBenchmark'] = safe_float(bench_sharpe_5y)
        
        # Align indices and create growth of $1,000 data
        growth_of_100 = []
        common_dates = portfolio_cum.index.intersection(benchmark_cum.index)
        
        for date in common_dates:
            growth_of_100.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio': safe_float(portfolio_cum[date]),
                'benchmark': safe_float(benchmark_cum[date])
            })
        
        # Drawdown series (portfolio vs benchmark)
        portfolio_drawdown = compute_drawdown_series(portfolio_cum)
        benchmark_drawdown = compute_drawdown_series(benchmark_cum)
        
        # Align drawdown series on common dates
        drawdown_common_dates = portfolio_drawdown.index.intersection(benchmark_drawdown.index)
        drawdown_data = []
        for date in drawdown_common_dates:
            drawdown_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio': safe_float(portfolio_drawdown[date] * 100),  # Convert to percentage
                'benchmark': safe_float(benchmark_drawdown[date] * 100)  # Convert to percentage
            })
        
        # Rolling Sharpe ratio (6-month window)
        portfolio_daily = compute_daily_returns(prices_portfolio, weights=weights)
        benchmark_daily = compute_daily_returns(prices_benchmark, weights=None)
        
        portfolio_rolling_sharpe = compute_rolling_sharpe_ratio(portfolio_daily, window_months=6, risk_free_rate=risk_free_rate)
        benchmark_rolling_sharpe = compute_rolling_sharpe_ratio(benchmark_daily, window_months=6, risk_free_rate=risk_free_rate)
        
        # Align rolling Sharpe series on common dates
        # Filter out NaN values (periods before full window is available)
        rolling_sharpe_common_dates = portfolio_rolling_sharpe.index.intersection(benchmark_rolling_sharpe.index)
        rolling_sharpe_data = []
        for date in rolling_sharpe_common_dates:
            port_sharpe = portfolio_rolling_sharpe[date]
            bench_sharpe = benchmark_rolling_sharpe[date]
            # Only include points where both values are valid (not NaN)
            if (port_sharpe is not None and not pd.isna(port_sharpe) and 
                bench_sharpe is not None and not pd.isna(bench_sharpe)):
                try:
                    rolling_sharpe_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'portfolio': safe_float(port_sharpe),
                        'benchmark': safe_float(bench_sharpe)
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # Rolling Volatility (6-month window, 126 trading days)
        # Ensure both portfolio and benchmark use the same aligned date range for consistency
        # This matches the alignment logic used in compute_rolling_beta
        if not prices_portfolio.empty and not prices_benchmark.empty:
            # Use the same alignment logic as rolling beta
            common_end_date = min(prices_portfolio.index.max(), prices_benchmark.index.max())
            cutoff_date = common_end_date - pd.DateOffset(months=60)
            
            # Filter both to same date range
            portfolio_filtered = prices_portfolio[prices_portfolio.index >= cutoff_date].sort_index()
            benchmark_filtered = prices_benchmark[prices_benchmark.index >= cutoff_date].sort_index()
            
            # Align on common dates (inner join) to ensure both start on the same date
            # This matches the logic used in compute_rolling_beta
            aligned_prices = pd.concat([portfolio_filtered, benchmark_filtered], axis=1, join='inner')
            aligned_prices = aligned_prices.dropna(how='any')  # Remove any rows with missing data
            
            if len(aligned_prices) > 0:
                # Split back into portfolio and benchmark, now with aligned dates
                portfolio_cols = [col for col in portfolio_filtered.columns if col in aligned_prices.columns]
                benchmark_cols = [col for col in benchmark_filtered.columns if col in aligned_prices.columns]
                
                prices_portfolio_aligned = aligned_prices[portfolio_cols] if portfolio_cols else pd.DataFrame()
                prices_benchmark_aligned = aligned_prices[benchmark_cols] if benchmark_cols else pd.DataFrame()
                
                portfolio_rolling_vol = compute_rolling_volatility(prices_portfolio_aligned, weights=weights) if not prices_portfolio_aligned.empty else pd.Series(dtype=float)
                benchmark_rolling_vol = compute_rolling_volatility(prices_benchmark_aligned, weights=None) if not prices_benchmark_aligned.empty else pd.Series(dtype=float)
            else:
                portfolio_rolling_vol = pd.Series(dtype=float)
                benchmark_rolling_vol = pd.Series(dtype=float)
        else:
            portfolio_rolling_vol = compute_rolling_volatility(prices_portfolio, weights=weights) if not prices_portfolio.empty else pd.Series(dtype=float)
            benchmark_rolling_vol = compute_rolling_volatility(prices_benchmark, weights=None) if not prices_benchmark.empty else pd.Series(dtype=float)
        
        # Verify rolling volatility series end dates match common_end_date (if sufficient data exists)
        # Note: Rolling series may end earlier if insufficient data for full window
        # Both portfolio and benchmark now use the same common_end_date, ensuring consistent start dates
        if len(portfolio_rolling_vol) > 0 and len(benchmark_rolling_vol) > 0:
            portfolio_vol_end = portfolio_rolling_vol.index.max()
            benchmark_vol_end = benchmark_rolling_vol.index.max()
            # Both should end at the same date (or earlier if insufficient data for full window)
            # This is expected if there's insufficient data for the rolling window
            # The series ends earlier because dropna() removes periods before full window
            pass  # Log if needed: rolling series may end earlier due to min_periods requirement
        
        # Align rolling volatility series on common dates (only non-null values)
        rolling_vol_common_dates = portfolio_rolling_vol.index.intersection(benchmark_rolling_vol.index)
        rolling_volatility_data = []
        for date in rolling_vol_common_dates:
            port_vol = portfolio_rolling_vol[date]
            bench_vol = benchmark_rolling_vol[date]
            # Only include points where both values are valid (not NaN)
            if (port_vol is not None and not pd.isna(port_vol) and 
                bench_vol is not None and not pd.isna(bench_vol)):
                try:
                    rolling_volatility_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'portfolio': safe_float(port_vol * 100),  # Convert to percentage
                        'benchmark': safe_float(bench_vol * 100)  # Convert to percentage
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # Rolling Beta (6-month window, 126 trading days) vs SPY
        rolling_beta_series = compute_rolling_beta(prices_portfolio, prices_benchmark, portfolio_weights=weights, window_days=126)
        
        # Verify rolling beta series end date matches actual_end_date (if sufficient data exists)
        # Note: Rolling series may end earlier if insufficient data for full window
        if len(rolling_beta_series) > 0:
            rolling_beta_end = rolling_beta_series.index.max()
            if rolling_beta_end != actual_end_date:
                # This is expected if there's insufficient data for the rolling window
                # The series ends earlier because dropna() removes periods before full window
                pass  # Log if needed: rolling series may end earlier due to min_periods requirement
        
        # Format rolling beta data for frontend (only non-null values)
        rolling_beta_data = []
        for date in rolling_beta_series.index:
            beta_value = rolling_beta_series[date]
            if beta_value is not None and not pd.isna(beta_value):
                try:
                    rolling_beta_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'beta': safe_float(beta_value)
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # YTD contributions to return
        ytd_contributions_list = []
        try:
            from portfolio_tool.analytics import is_cash_ticker
            ytd_contributions = compute_ytd_contribution(prices_portfolio, weights=weights)
            # Filter out cash tickers from visualization
            for ticker, contribution in ytd_contributions.items():
                if not pd.isna(contribution) and not is_cash_ticker(ticker):
                    ytd_contributions_list.append({
                        'ticker': ticker,
                        'contribution': safe_float(contribution * 100)  # Convert to percentage points
                    })
        except (ValueError, Exception) as e:
            # If no YTD data is available, return empty list
            # This can happen if the current year just started or if there's insufficient data
            ytd_contributions_list = []
        
        # YTD risk contributions (percentage of total portfolio risk)
        ytd_risk_contributions_list = []
        try:
            from portfolio_tool.analytics import is_cash_ticker
            ytd_risk_contributions = compute_ytd_risk_contribution(prices_portfolio, weights=weights)
            # Filter out cash tickers from visualization
            for ticker, contribution in ytd_risk_contributions.items():
                if not pd.isna(contribution) and not is_cash_ticker(ticker):
                    ytd_risk_contributions_list.append({
                        'ticker': ticker,
                        'contribution': safe_float(contribution * 100)  # Convert to percentage (already normalized)
                    })
            # Renormalize to sum to 100% across non-cash holdings
            if ytd_risk_contributions_list:
                total = sum(c['contribution'] for c in ytd_risk_contributions_list)
                if total > 0:
                    for contrib in ytd_risk_contributions_list:
                        contrib['contribution'] = contrib['contribution'] / total * 100
        except (ValueError, Exception) as e:
            # If no YTD data is available, return empty list
            ytd_risk_contributions_list = []
        
        # Monthly returns heatmap (last 5 years)
        monthly_portfolio = compute_monthly_portfolio_returns(
            prices_portfolio, weights=weights, years_back=5
        )
        
        # Format monthly returns heatmap
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years_list = sorted(monthly_portfolio.index.tolist(), reverse=True)
        
        heatmap_values = []
        for year in years_list:
            for month_idx, month_name in enumerate(months, start=1):
                if month_name in monthly_portfolio.columns:
                    return_val = monthly_portfolio.loc[year, month_name]
                    if pd.notna(return_val):
                        heatmap_values.append({
                            'year': int(year),
                            'month': month_idx,
                            'return': safe_float(return_val)
                        })
        
        # Correlation matrix
        correlation_matrix = compute_correlation_matrix(prices_portfolio, years=3)
        
        # Risk-return scatter
        benchmark_tickers = ['SPY', 'QQQ', 'AGG', 'ACWI']
        all_benchmark_tickers = [benchmark_ticker] + [t for t in benchmark_tickers if t != benchmark_ticker]
        benchmark_prices_all = get_price_history(all_benchmark_tickers, start_date, end_date)
        
        risk_return_scatter = []
        
        # Portfolio
        port_ret, port_vol = compute_annualized_return_and_volatility(
            prices_portfolio, weights=weights, years=5
        )
        if port_ret is not None and port_vol is not None:
            risk_return_scatter.append({
                'label': 'Portfolio',
                'return': safe_float(port_ret),
                'risk': safe_float(port_vol)
            })
        
        # SPY
        spy_ret, spy_vol = compute_annualized_return_and_volatility(
            prices_benchmark, weights=None, years=5
        )
        if spy_ret is not None and spy_vol is not None:
            risk_return_scatter.append({
                'label': 'SPY',
                'return': safe_float(spy_ret),
                'risk': safe_float(spy_vol)
            })
        
        # Other benchmarks (exclude SPY since it's already added above)
        for ticker in benchmark_tickers:
            if ticker == benchmark_ticker:  # Skip SPY - already added above
                continue
            if ticker in benchmark_prices_all.columns:
                ticker_prices = benchmark_prices_all[[ticker]]
                # Filter to effective_start_date for consistency with portfolio and SPY
                if effective_start_date is not None:
                    ticker_prices = ticker_prices[ticker_prices.index >= effective_start_date]
                ticker_ret, ticker_vol = compute_annualized_return_and_volatility(
                    ticker_prices, weights=None, years=5
                )
                if ticker_ret is not None and ticker_vol is not None:
                    risk_return_scatter.append({
                        'label': ticker,
                        'return': safe_float(ticker_ret),
                        'risk': safe_float(ticker_vol)
                    })
        
        # Efficient frontier
        benchmark_prices_dict = {benchmark_ticker: prices_benchmark}
        for ticker in benchmark_tickers:
            if ticker in benchmark_prices_all.columns:
                benchmark_prices_dict[ticker] = benchmark_prices_all[[ticker]]
        
        ef_data = compute_efficient_frontier_analysis(
            portfolio_prices=prices_portfolio,
            portfolio_weights=weights,
            benchmark_prices_dict=benchmark_prices_dict,
            years=5
        )
        
        # Format efficient frontier
        efficient_frontier = {
            'points': [],
            'current': {'risk': 0.0, 'return': 0.0},
            'maxSharpe': {'risk': 0.0, 'return': 0.0},
            'minVariance': {'risk': 0.0, 'return': 0.0}
        }
        
        if ef_data and ef_data.get('frontier') is not None:
            frontier_df = ef_data['frontier']
            for _, row in frontier_df.iterrows():
                efficient_frontier['points'].append({
                    'risk': safe_float(row['vol']),
                    'return': safe_float(row['ret'])
                })
            
            # Current portfolio point - use computed portfolio metrics
            if port_ret is not None and port_vol is not None:
                efficient_frontier['current'] = {
                    'risk': safe_float(port_vol),
                    'return': safe_float(port_ret)
                }
            elif ef_data.get('portfolio_point'):
                portfolio_point = ef_data['portfolio_point']
                if isinstance(portfolio_point, dict):
                    efficient_frontier['current'] = {
                        'risk': safe_float(portfolio_point.get('vol', 0.0)),
                        'return': safe_float(portfolio_point.get('ret', 0.0))
                    }
            
            # Max Sharpe (tangency portfolio)
            if ef_data.get('tangency'):
                tangency = ef_data['tangency']
                efficient_frontier['maxSharpe'] = {
                    'risk': safe_float(tangency.get('vol', 0.0)),
                    'return': safe_float(tangency.get('ret', 0.0))
                }
            
            # Min variance - find point with lowest risk in frontier
            if efficient_frontier['points']:
                min_var_point = min(efficient_frontier['points'], key=lambda x: x['risk'])
                efficient_frontier['minVariance'] = min_var_point
        
        # Holdings
        holdings = [
            {'ticker': ticker, 'weight': safe_float(weight)}
            for ticker, weight in weights.items()
        ]
        
        # Warnings
        warnings = []
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.05:  # 5% tolerance
            warnings.append(f'Portfolio weights sum to {total_weight*100:.1f}%, expected 100%')
        
        # Build response matching PortfolioDashboardResponse
        # Use actual_end_date (last date in data) instead of today for consistency with displayed end date
        response_data = {
            'meta': {
                'analysisDate': actual_end_date.isoformat() if isinstance(actual_end_date, pd.Timestamp) else pd.Timestamp(actual_end_date).isoformat(),
                'benchmarkTicker': benchmark_ticker,
                'riskFreeRate': safe_float(risk_free_rate),
                'lookbackYears': 5,
                'effectiveStartDate': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None
            },
            'periodReturns': {
                'portfolio': period_returns_portfolio,
                'benchmark': period_returns_benchmark,
                'annualized': True
            },
            'returnsTable': returns_table,
            'riskMetrics': risk_metrics,
            'benchmarkRiskMetrics': benchmark_risk_metrics,
            'performanceMetrics5Y': performance_metrics_5y,
            'charts': {
                'growthOf100': growth_of_100,
                'drawdown': drawdown_data,
                'rollingSharpe': rolling_sharpe_data,
                'rollingVolatility': rolling_volatility_data,
                'rollingBeta': rolling_beta_data,
                'ytdContributions': ytd_contributions_list,
                'ytdRiskContributions': ytd_risk_contributions_list,
                'monthlyReturnsHeatmap': {
                    'years': years_list,
                    'months': months,
                    'values': heatmap_values
                },
                'correlationMatrix': {
                    'tickers': correlation_matrix.columns.tolist(),
                    'matrix': [[safe_float(val) for val in row] for row in correlation_matrix.values.tolist()]
                },
                'riskReturnScatter': risk_return_scatter,
                'efficientFrontier': efficient_frontier
            },
            'holdings': holdings,
            'warnings': warnings
        }
        
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Portfolio analysis error: {error_detail}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "portfolio-analysis-api"}


if __name__ == "__main__":
    import uvicorn
    # Note: For auto-reload during development, use from command line:
    # uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
    # Running directly with reload=False avoids the warning
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)

