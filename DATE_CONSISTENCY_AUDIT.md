# Date Consistency Audit Report
**Date:** 2025-12-29  
**Scope:** Full analytics pipeline start-date and end-date consistency

## Executive Summary

**Status:** ❌ **INCONSISTENCIES FOUND**

Two functions use `datetime.today()` instead of the actual end date from data:
1. `compute_rolling_volatility()` - Line 613
2. `compute_rolling_beta()` - Line 847

All other calculations correctly use the actual end date from the dataset.

---

## Source of Truth

**Effective Start Date:** `effective_start_date = max(latest_first_valid_date, default_60_months_back)`
- Computed in: `get_effective_start_date()` (analytics.py:69-113)
- Applied to: Both portfolio and benchmark prices in api_server.py:182-185

**Actual End Date:** `actual_end_date = get_as_of_date(prices.index)` 
- Computed in: api_server.py:159-164
- Used for: All period calculations, API response `analysisDate`

---

## Detailed Audit Checklist

### ✅ Backend — Data & Analytics

#### Price Fetching (`get_price_history`)
- **Start date:** `today - 6 years` (api_server.py:139)
- **End date:** `today + 1 day` (api_server.py:141) - accounts for yfinance exclusive end parameter
- **Resulting DataFrame:** `index.min()` and `index.max()` from actual data
- **Status:** ✅ Correct - uses tomorrow to ensure today's data is included

#### Effective Start Date Logic (`get_effective_start_date`)
- **Computation:** `max(latest_first_valid_date, default_60_months_back)` (analytics.py:111)
- **Applied to:** Both portfolio and benchmark (api_server.py:182-185)
- **Status:** ✅ Correct - explicitly computed and applied consistently

#### Daily Returns Construction (`compute_daily_returns`)
- **Start:** Uses prices already filtered to `effective_start_date` (api_server.py:182-185)
- **End:** Uses actual last date in prices DataFrame
- **Status:** ✅ Correct - prices are pre-filtered, function uses all available dates

#### Period Returns (`compute_period_returns`)
- **Cutoffs:** `cutoff_date = daily_returns.index[-1] - pd.DateOffset(...)` (analytics.py:240, 244)
- **YTD:** Uses `as_of = get_as_of_date(daily_returns.index)` (analytics.py:234)
- **No datetime.today():** ✅ Confirmed - uses `daily_returns.index[-1]` or `get_as_of_date()`
- **Status:** ✅ Correct - all periods computed from actual data end date

#### CAGR / 3Y / 5Y Availability Checks
- **History length:** `history_years = (end_date - start_date).days / 365.25` (analytics.py:201)
- **Where:** `start_date = prices.index[0]`, `end_date = prices.index[-1]` (analytics.py:199-200)
- **Insufficient history:** Returns `None` if `history_years < (required_years * 0.95)` (analytics.py:226)
- **Status:** ✅ Correct - uses actual date range from filtered prices

#### Risk Metrics (volatility, Sharpe, Sortino, beta, drawdown)
- **Volatility:** Uses `daily_returns.index[-1] - pd.DateOffset(years=years)` (analytics.py:578)
- **Sharpe:** Uses `daily_returns.index[-1] - pd.DateOffset(years=years)` (analytics.py:666)
- **Sortino:** Uses `daily_returns.index[-1] - pd.DateOffset(years=years)` (analytics.py:703)
- **Max Drawdown:** Uses `cumulative_index.index[-1] - pd.DateOffset(years=years)` (analytics.py:948)
- **Status:** ✅ Correct - all use actual end date from data

#### ❌ Rolling Volatility (`compute_rolling_volatility`)
- **Location:** analytics.py:610-650
- **Issue:** Line 613 uses `today = datetime.today()` instead of actual end date
- **Current:** `cutoff_date = today - pd.DateOffset(months=60)` (line 614)
- **Should be:** `cutoff_date = prices.index[-1] - pd.DateOffset(months=60)`
- **Impact:** Rolling volatility chart may show data ending on a different date than other metrics
- **Status:** ❌ **INCONSISTENT**

#### ❌ Rolling Beta (`compute_rolling_beta`)
- **Location:** analytics.py:842-920
- **Issue:** Line 847 uses `today = datetime.today()` instead of actual end date
- **Current:** `cutoff_date = today - pd.DateOffset(months=60)` (line 848)
- **Should be:** `cutoff_date = portfolio_prices.index[-1] - pd.DateOffset(months=60)`
- **Impact:** Rolling beta chart may show data ending on a different date than other metrics
- **Status:** ❌ **INCONSISTENT**

#### Risk-Return Scatterplot
- **Function:** `compute_annualized_return_and_volatility()` (analytics.py:1084-1126)
- **Cutoff:** `cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)` (line 1100)
- **Portfolio/Benchmark alignment:** Both use same function with same `years` parameter
- **Status:** ✅ Correct - uses actual end date, portfolio and benchmark aligned

#### Correlation Matrix
- **Function:** `compute_correlation_matrix()` (analytics.py:1065-1082)
- **Cutoff:** `cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)` (line 1076)
- **Status:** ✅ Correct - uses actual end date

#### Efficient Frontier
- **Function:** `compute_efficient_frontier_analysis()` (analytics.py:1356-1420)
- **Cutoff:** `cutoff_date = portfolio_daily.index[-1] - pd.DateOffset(years=years)` (line 1379)
- **Status:** ✅ Correct - uses actual end date

#### Monthly Returns Heatmap
- **Function:** `compute_monthly_portfolio_returns()` (analytics.py:442-471)
- **YTD year:** Uses `as_of_date = get_as_of_date(prices.index)` (line 448)
- **Status:** ✅ Correct - uses actual end date

#### Yearly Returns (`compute_returns`)
- **YTD year:** Uses `as_of_date = get_as_of_date(prices.index)` (analytics.py:155)
- **Status:** ✅ Correct - uses actual end date

### ✅ API Layer

#### API Response Fields
- **analysisDate:** `actual_end_date.isoformat()` (api_server.py:669)
- **effectiveStartDate:** `effective_start_date.strftime('%Y-%m-%d')` (api_server.py:673)
- **Status:** ✅ Correct - both use computed dates from data

#### No Implicit Dates
- **datetime.today() usage:** Only in api_server.py:138 (for initial fetch range) and in the two rolling functions (issues above)
- **Status:** ✅ Mostly correct - only rolling functions have issues

### ✅ Frontend

#### Displayed Start and End Dates
- **Start Date:** Uses `analysisData.meta.effectiveStartDate` (PortfolioResults.tsx:1573)
- **End Date:** Uses last date from `analysisData.charts.growthOf100` (PortfolioResults.tsx:1584)
- **Status:** ✅ Correct - uses API-provided dates, no recomputation

#### Charts
- **Growth of $1,000:** Uses `common_dates = portfolio_cum.index.intersection(benchmark_cum.index)` (api_server.py:383)
- **All charts:** Bounded by `effective_start_date → actual_end_date` after filtering
- **Status:** ✅ Correct - all use filtered price data

#### Summary Cards & AI Insights
- **Data source:** All metrics come from API response computed using same date window
- **Status:** ✅ Correct - frontend displays API data, no local computation

---

## Issues Found

### Issue #1: `compute_rolling_volatility()` uses `datetime.today()`

**File:** `portfolio_tool/analytics.py`  
**Lines:** 610-650  
**Problem:** Line 613 uses `today = datetime.today()` instead of the actual end date from the data.

**Current Code:**
```python
def compute_rolling_volatility(prices, weights=None):
    from datetime import datetime
    
    # Filter to last 5 years (60 months) ending today
    today = datetime.today()  # ❌ WRONG
    cutoff_date = today - pd.DateOffset(months=60)
    prices_filtered = prices[prices.index >= cutoff_date].sort_index()
```

**Fix:**
```python
def compute_rolling_volatility(prices, weights=None):
    # Filter to last 5 years (60 months) ending at actual data end date
    actual_end_date = prices.index[-1]  # ✅ Use actual end date
    cutoff_date = actual_end_date - pd.DateOffset(months=60)
    prices_filtered = prices[prices.index >= cutoff_date].sort_index()
```

**Impact:** Rolling volatility chart may end on a different date than other metrics if run on weekends/holidays or when yfinance data is delayed.

---

### Issue #2: `compute_rolling_beta()` uses `datetime.today()`

**File:** `portfolio_tool/analytics.py`  
**Lines:** 842-920  
**Problem:** Line 847 uses `today = datetime.today()` instead of the actual end date from the data.

**Current Code:**
```python
def compute_rolling_beta(portfolio_prices, benchmark_prices, portfolio_weights, window_days=126):
    from datetime import datetime
    
    # Filter both price DataFrames to the last 5 years ending today
    today = datetime.today()  # ❌ WRONG
    cutoff_date = today - pd.DateOffset(months=60)
    
    portfolio_filtered = portfolio_prices[portfolio_prices.index >= cutoff_date].sort_index()
    benchmark_filtered = benchmark_prices[benchmark_prices.index >= cutoff_date].sort_index()
```

**Fix:**
```python
def compute_rolling_beta(portfolio_prices, benchmark_prices, portfolio_weights, window_days=126):
    # Filter both price DataFrames to the last 5 years ending at actual data end date
    # Use the minimum of the two end dates to ensure alignment
    actual_end_date = min(portfolio_prices.index[-1], benchmark_prices.index[-1])  # ✅ Use actual end date
    cutoff_date = actual_end_date - pd.DateOffset(months=60)
    
    portfolio_filtered = portfolio_prices[portfolio_prices.index >= cutoff_date].sort_index()
    benchmark_filtered = benchmark_prices[benchmark_prices.index >= cutoff_date].sort_index()
```

**Impact:** Rolling beta chart may end on a different date than other metrics if run on weekends/holidays or when yfinance data is delayed.

---

## Summary

**Total Functions Audited:** 15  
**Functions with Issues:** 2  
**Functions Correct:** 13

**Overall Status:** The platform is **mostly consistent**, with only two functions (`compute_rolling_volatility` and `compute_rolling_beta`) using `datetime.today()` instead of the actual end date from the data.

**Recommendation:** Fix the two rolling functions to use the actual end date from the data for complete consistency.

