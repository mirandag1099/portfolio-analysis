import pandas as pd
import yfinance as yf

def _map_etf_category(ticker):
    """
    Map common ETFs to user-friendly industry categories.
    This is needed because yfinance doesn't provide good sector/industry info for ETFs.
    
    Returns:
        str: Industry category for the ETF, or None if not an ETF we recognize
    """
    ticker_upper = ticker.upper()
    
    # Comprehensive ETF mapping
    etf_mapping = {
        # U.S. Large Cap Equity
        'SPY': 'U.S. Large Cap Equity',
        'VOO': 'U.S. Large Cap Equity',
        'IVV': 'U.S. Large Cap Equity',
        'SPLG': 'U.S. Large Cap Equity',
        
        # U.S. Mid Cap Equity
        'MDY': 'U.S. Mid Cap Equity',
        'VO': 'U.S. Mid Cap Equity',
        'IJH': 'U.S. Mid Cap Equity',
        
        # U.S. Small Cap Equity
        'IWM': 'U.S. Small Cap Equity',
        'VB': 'U.S. Small Cap Equity',
        'IJR': 'U.S. Small Cap Equity',
        
        # U.S. Total Market Equity
        'VTI': 'U.S. Total Market Equity',
        'ITOT': 'U.S. Total Market Equity',
        'SWTSX': 'U.S. Total Market Equity',
        
        # Growth ETFs
        'QQQ': 'U.S. Growth Equity',
        'VUG': 'U.S. Growth Equity',
        'IWF': 'U.S. Growth Equity',
        'VGT': 'U.S. Technology Equity',
        
        # Value ETFs
        'VTV': 'U.S. Value Equity',
        'IWD': 'U.S. Value Equity',
        'VYM': 'U.S. Value Equity',
        
        # Dividend ETFs
        'SCHD': 'U.S. Dividend Equity',
        'VIG': 'U.S. Dividend Equity',
        'DVY': 'U.S. Dividend Equity',
        
        # Sector-specific U.S. Equity
        'XLK': 'U.S. Technology Equity',
        'XLF': 'U.S. Financial Equity',
        'XLE': 'U.S. Energy Equity',
        'XLV': 'U.S. Healthcare Equity',
        'XLI': 'U.S. Industrial Equity',
        'XLP': 'U.S. Consumer Staples Equity',
        'XLY': 'U.S. Consumer Discretionary Equity',
        'XLB': 'U.S. Materials Equity',
        'XLU': 'U.S. Utilities Equity',
        'XLRE': 'U.S. Real Estate Equity',
        'XLC': 'U.S. Communication Equity',
        
        # International Equity
        'VEA': 'Developed International Equity',
        'EFA': 'Developed International Equity',
        'VXUS': 'International Equity',
        'ACWI': 'Global Equity',
        'VT': 'Global Equity',
        'VWO': 'Emerging Markets Equity',
        'EEM': 'Emerging Markets Equity',
        'VXUS': 'International Equity',
        
        # Regional International Equity
        'VPL': 'Asia-Pacific Equity',
        'VGK': 'European Equity',
        'EWJ': 'Japan Equity',
        'FXI': 'China Equity',
        
        # Fixed Income / Bonds
        'BND': 'U.S. Total Bond Market',
        'AGG': 'U.S. Investment Grade Bonds',
        'VGLT': 'U.S. Long-Term Treasury',
        'VGIT': 'U.S. Intermediate-Term Treasury',
        'VGSH': 'U.S. Short-Term Treasury',
        'VCLT': 'U.S. Long-Term Corporate Bonds',
        'VCIT': 'U.S. Intermediate-Term Corporate Bonds',
        'VCSH': 'U.S. Short-Term Corporate Bonds',
        'TLT': 'U.S. Long-Term Treasury',
        'IEF': 'U.S. Intermediate-Term Treasury',
        'SHY': 'U.S. Short-Term Treasury',
        'LQD': 'U.S. Corporate Bonds',
        'HYG': 'U.S. High Yield Bonds',
        'JNK': 'U.S. High Yield Bonds',
        'MUB': 'U.S. Municipal Bonds',
        'EMB': 'Emerging Markets Bonds',
        
        # Real Estate
        'VNQ': 'U.S. Real Estate',
        'SCHH': 'U.S. Real Estate',
        'IYR': 'U.S. Real Estate',
        
        # Commodities
        # Gold
        'GLD': 'Gold',
        'GLDM': 'Gold',
        'IAU': 'Gold',
        'SGOL': 'Gold',
        'OUNZ': 'Gold',
        # Silver
        'SLV': 'Silver',
        'SIVR': 'Silver',
        'PSLV': 'Silver',
        # Platinum & Palladium
        'PPLT': 'Platinum',
        'PLTM': 'Platinum',
        'PALL': 'Palladium',
        # Broad Commodities
        'DBC': 'Broad Commodities',
        'DJP': 'Broad Commodities',
        'PDBC': 'Broad Commodities',
        'USCI': 'Broad Commodities',
        'GSG': 'Broad Commodities',
        # Energy / Oil
        'USO': 'Oil',
        'DBO': 'Oil',
        'UCO': 'Oil',
        'BNO': 'Brent Oil',
        'DBE': 'Energy',
        # Natural Gas
        'UNG': 'Natural Gas',
        'BOIL': 'Natural Gas',
        'KOLD': 'Natural Gas',
        # Agriculture
        'DBA': 'Agriculture',
        'CORN': 'Corn',
        'SOYB': 'Soybeans',
        'WEAT': 'Wheat',
        'NIB': 'Cocoa',
        'CANE': 'Sugar',
        'JO': 'Coffee',
        
        # Alternative / Other
        'TIP': 'U.S. Inflation-Protected Bonds',
        'SHV': 'U.S. Short-Term Treasury',
    }
    
    return etf_mapping.get(ticker_upper)

def _get_etf_sector(industry):
    """
    Map ETF industry categories to broader sector categories for grouping.
    """
    if 'Equity' in industry:
        if 'U.S.' in industry:
            return 'U.S. Equity'
        elif 'International' in industry or 'Global' in industry or 'Emerging' in industry:
            return 'International Equity'
        else:
            return 'Equity'
    elif 'Bond' in industry or 'Treasury' in industry or 'Fixed Income' in industry:
        return 'Fixed Income'
    elif 'Real Estate' in industry:
        return 'Real Estate'
    elif ('Gold' in industry or 'Silver' in industry or 'Platinum' in industry or 
          'Palladium' in industry or 'Commodities' in industry or 'Agriculture' in industry or
          'Oil' in industry or 'Energy' in industry or 'Natural Gas' in industry or
          'Corn' in industry or 'Soybeans' in industry or 'Wheat' in industry or
          'Cocoa' in industry or 'Sugar' in industry or 'Coffee' in industry or
          'Brent' in industry):
        return 'Commodities'
    else:
        return 'Other'

def get_price_history(tickers, start_date, end_date):
    """
    Download daily prices for the given tickers from yfinance.
    Uses auto-adjusted Close prices.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,  # use adjusted prices
        progress=False
    )

    # yfinance returns a multi-index DataFrame when multiple tickers are provided
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data

    # ensure columns are exactly the tickers list (and in the same order)
    prices = prices[tickers]

    # drop rows with all NaNs (non-trading days)
    prices = prices.dropna(how='all')

    return prices

def get_sector_info(tickers):
    """
    Fetch sector and industry information for given tickers using yfinance.
    For stocks: uses sector from yfinance API.
    For ETFs: uses smart mapping to user-friendly categories, then groups into broader sectors.
    
    Returns:
        DataFrame with columns: ticker, sector, industry
    """
    sector_data = []
    
    for ticker in tickers:
        # Check if this is an ETF we recognize
        etf_industry = _map_etf_category(ticker)
        
        if etf_industry is not None:
            # This is an ETF - use smart mapping
            etf_sector = _get_etf_sector(etf_industry)
            sector_data.append({
                'ticker': ticker,
                'sector': etf_sector,
                'industry': etf_industry
            })
        else:
            # This is likely a stock - try to get sector/industry from yfinance
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                # If both are Unknown, it might be an ETF we don't recognize
                if sector == 'Unknown' and industry == 'Unknown':
                    # Try to detect if it's an ETF by checking quoteType
                    quote_type = info.get('quoteType', '').upper()
                    if quote_type == 'ETF':
                        # Unrecognized ETF - mark as such
                        sector_data.append({
                            'ticker': ticker,
                            'sector': 'Unknown',
                            'industry': 'Unknown ETF'
                        })
                    else:
                        sector_data.append({
                            'ticker': ticker,
                            'sector': sector,
                            'industry': industry
                        })
                else:
                    sector_data.append({
                        'ticker': ticker,
                        'sector': sector,
                        'industry': industry
                    })
            except Exception as e:
                # If we can't fetch data for a ticker, mark as Unknown
                sector_data.append({
                    'ticker': ticker,
                    'sector': 'Unknown',
                    'industry': 'Unknown'
                })
    
    return pd.DataFrame(sector_data)

def get_risk_free_rate(ticker='^TNX'):
    """
    Fetch the current risk-free rate (default: 10-year Treasury yield ^TNX).
    
    ticker: ticker symbol for risk-free rate (default: '^TNX' for 10-year Treasury)
    
    Returns: annual risk-free rate as decimal (e.g., 0.04 for 4%)
             Returns None if data cannot be fetched
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Try to get the current yield from info
        current_yield = info.get('regularMarketPrice')
        
        if current_yield is None:
            # Fallback: try to get from recent price data
            hist = ticker_obj.history(period='5d')
            if not hist.empty:
                current_yield = hist['Close'].iloc[-1]
        
        if current_yield is not None:
            # Convert percentage to decimal (^TNX is quoted as percentage, e.g., 4.5 means 4.5%)
            return float(current_yield) / 100.0
        
        return None
    except Exception as e:
        # If we can't fetch the risk-free rate, return None
        # The calling code can handle this (e.g., use 0.0 as fallback)
        return None
