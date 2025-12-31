# Portfolio Analysis Tool

A comprehensive Python-based investment portfolio analysis tool that analyzes your investment portfolio based on a CSV file containing tickers and weights. Using historical market data, it computes performance metrics, risk insights, and visualizations to help you understand your portfolio's behavior.

## Features

- **Performance Analytics**
  - Calendar-year and year-to-date (YTD) returns
  - Period returns (1M, 3M, 6M, 1Y, YTD, 3Y, 5Y)
  - Cumulative returns tracking
  - Monthly returns analysis
  - Portfolio vs. Benchmark comparison (S&P 500)

- **Risk Analytics**
  - Volatility (standard deviation)
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Beta (portfolio sensitivity to market)
  - Correlation matrix between assets
  - Risk-Return scatter plot

- **Portfolio Optimization**
  - Efficient Frontier analysis
  - Tangency Portfolio (optimal risk-adjusted return)
  - Random portfolio generation for comparison

- **Portfolio Insights**
  - YTD contribution analysis by asset
  - Sector/Industry allocation breakdown
  - Comprehensive ETF classification (including commodities, bonds, real estate)
  - Benchmark comparison (S&P 500)

- **Visualizations**
  - Cumulative returns chart (Portfolio vs. Benchmark)
  - Return contributions bar chart
  - Monthly returns heatmap
  - Sector allocation pie chart
  - Correlation matrix heatmap
  - Risk-Return scatter plot
  - Efficient Frontier visualization

- **Robust CSV Handling**
  - Flexible column names (ticker/symbol/stock, weight/allocation/percentage)
  - Multiple weight formats (percentages: "20%", decimals: "0.20", or raw: "20")
  - Automatic encoding detection (UTF-8, Latin-1, CP1252)
  - Handles BOM characters, whitespace, and empty rows
  - Automatic weight normalization
  - Duplicate ticker aggregation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Portfolio Analysis"
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy matplotlib yfinance scipy
```

## Usage

### 1. Prepare Your Portfolio CSV

Create a CSV file (e.g., `my_portfolio.csv`) with ticker symbols and weights. The tool is flexible and accepts various formats:

**Standard Format:**
```csv
ticker,weight
AAPL,0.25
MSFT,0.25
AMZN,0.20
NVDA,0.15
TSLA,0.15
```

**Flexible Column Names** (all accepted):
- Ticker columns: `ticker`, `symbol`, `stock`, `ticker_symbol`
- Weight columns: `weight`, `allocation`, `percentage`, `percent`, `pct`, `weight_pct`

**Flexible Weight Formats** (all accepted):
- Decimals: `0.25`, `0.20`
- Percentages: `25%`, `20%`
- Raw percentages: `25`, `20` (values > 1 are treated as percentages)

**Example with percentages:**
```csv
Symbol,Allocation
AAPL,25%
MSFT,25%
AMZN,20%
NVDA,15%
TSLA,15%
```

**Note:** 
- Weights will be automatically normalized to sum to 1.0 if they don't already
- Case-insensitive column names
- Extra columns are ignored
- Duplicate tickers are automatically aggregated
- Empty rows and invalid entries are automatically filtered out

### 2. Run the Analysis

**Basic usage:**
```bash
python main.py
```

**Specify a custom portfolio file:**
```bash
python main.py my_portfolio.csv
```

The script will:
- Load your portfolio from the CSV file (with automatic format detection and cleaning)
- Fetch historical price data (6 years)
- Compute performance metrics (returns, contributions)
- Compute risk metrics (volatility, Sharpe, Sortino, drawdown, beta)
- Generate efficient frontier analysis
- Display results in the console
- Generate interactive visualizations

## Project Structure

```
Portfolio Analysis/
├── main.py                 # Entry point - orchestrates the workflow
├── my_portfolio.csv        # Example portfolio CSV file
├── portfolio_tool/
│   ├── __init__.py
│   ├── data_io.py          # CSV loading and validation
│   ├── market_data.py      # yfinance integration for price/sector data
│   ├── analytics.py        # Core analytics functions
│   ├── reporting.py        # CLI text output and summaries
│   └── plots.py            # Visualization functions
└── README.md
```

## Architecture

The project follows a modular architecture:

- **`main.py`**: Clean orchestration layer - no analytics logic
- **`portfolio_tool/data_io.py`**: Data loading and validation
- **`portfolio_tool/market_data.py`**: External data fetching (yfinance)
- **`portfolio_tool/analytics.py`**: All computation logic (pure functions)
- **`portfolio_tool/reporting.py`**: CLI text output and narrative summaries
- **`portfolio_tool/plots.py`**: All visualization logic

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **yfinance**: Yahoo Finance API for market data
- **scipy**: Scientific computing (for optimization in efficient frontier)

## Example Output

The tool generates comprehensive analysis including:

**Console Output:**
- Yearly returns (by calendar year and YTD)
- Period returns (1M, 3M, 6M, 1Y, YTD, 3Y, 5Y)
- YTD contribution summary by asset
- Risk metrics table (volatility, Sharpe, Sortino, max drawdown, beta)
- Portfolio vs. Benchmark comparison

**Interactive Visualizations:**
- Cumulative returns comparison (Portfolio vs. S&P 500)
- Return contributions breakdown by asset
- Monthly returns heatmap
- Sector allocation pie chart (with comprehensive ETF classification)
- Correlation matrix heatmap
- Risk-Return scatter plot
- Efficient Frontier with tangency portfolio

## ETF Classification

The tool includes comprehensive ETF mappings for accurate sector classification:

- **U.S. Equity ETFs**: VOO, SPY, QQQ, VTI, VEA, VWO, and many more
- **International Equity ETFs**: VXUS, EFA, EEM, IEMG, and others
- **Fixed Income ETFs**: AGG, BND, TLT, VCLT, VCIT, VCSH, and more
- **Real Estate ETFs**: VNQ, SCHH, IYR
- **Commodity ETFs**: 
  - Gold: GLD, GLDM, IAU, SGOL
  - Silver: SLV, SIVR, PSLV
  - Energy: USO, DBO, UCO, BNO, UNG
  - Agriculture: DBA, CORN, SOYB, WEAT
  - Broad Commodities: DBC, DJP, PDBC, USCI, GSG
- **Alternative Assets**: TIP, SHV, and others

ETFs are automatically classified into appropriate sectors for the allocation pie chart.

## Future Enhancements

Planned features include:
- FastAPI backend for web UI integration
- Exportable PDF reports
- Additional optimization strategies
- Multi-currency support
- Custom benchmark selection

## Contributing

This project follows a clean, modular architecture. When adding features:
- Keep analytics logic in `analytics.py`
- Keep visualization logic in `plots.py`
- Keep CLI text in `reporting.py`
- Keep data I/O logic in `data_io.py`
- Keep market data fetching in `market_data.py`
- Maintain `main.py` as a clean orchestration layer
- Use pure functions when possible
- Add new ETF mappings to `market_data.py` when needed

## Known Limitations

- Historical data is limited to what's available via yfinance (typically 6+ years)
- Some delisted or very new tickers may not have sufficient data
- ETF classification is based on a curated mapping list (unknown ETFs will show as "Unknown" in sector breakdown)
- Benchmark is currently fixed to S&P 500 (^GSPC)

## License

[Add your license here]

## Author

[Add your name/contact information here]

