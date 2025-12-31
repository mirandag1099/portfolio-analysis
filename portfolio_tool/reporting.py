import pandas as pd

def format_pct(df: pd.DataFrame) -> pd.DataFrame:
    return (df * 100).round(2).astype(str) + '%'


def display_yearly_results(ticker_returns, portfolio_returns, benchmark_returns):
    print("\nTicker Yearly Returns:")
    print(format_pct(ticker_returns))

    comparison_df = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'Benchmark': benchmark_returns
    })

    comparison_pct = format_pct(comparison_df)
    print("\nPortfolio vs Benchmark Yearly Returns:")
    print(comparison_pct)


def display_period_results(portfolio_period: dict, benchmark_period: dict):
    comparison_period = pd.DataFrame(
        [portfolio_period, benchmark_period],
        index=['Portfolio', 'Benchmark - S&P 500']
    )

    comparison_period_pct = format_pct(comparison_period)
    print("\nPeriod Returns:")
    print(comparison_period_pct)

def summarize_ytd_contributions(contributions, top_n: int = 3):
    """
    Print a short text summary of YTD portfolio returns and top contributors.
    
    Filters out cash from top contributors but includes cash in total.

    contributions: Series indexed by ticker
    """
    from portfolio_tool.analytics import is_cash_ticker
    
    contributions = contributions.dropna()
    if contributions.empty:
        print("No contributions available for summary.")
        return
    
    # Total includes cash (cash contributes 0.0)
    total = contributions.sum()
    total_pct = total * 100

    print("\nYTD Portfolio Return Summary:")

    # Handle near zero total return
    if abs(total) < 1e-4:
        print("The portfolio's YTD return is approximately 0.00%.")
        return
    
    # Filter out cash from top contributors
    non_cash_contributions = contributions[[ticker for ticker in contributions.index 
                                          if not is_cash_ticker(ticker)]]
    
    # Top N positive contributors (excluding cash)
    top = non_cash_contributions.sort_values(ascending=False).head(top_n)
    top_pct = top * 100

    # build a readable list:
    top_str = ", ".join(
        f"{ticker} ({val:+.2f}pp)" for ticker, val in top_pct.items()
    )

    # Share of total result explained by top N (use absolute total to avoid sign weirdness)
    explained_share = (top_pct.sum() / total_pct) * 100 if total_pct != 0 else None

    direction = "gain" if total > 0 else "loss"

    print(f"Portfolio YTD return: {total_pct:+.2f} percentage points ({direction}).")
    print(f"Top {top_n} contributors: {top_str}.")
    if explained_share is not None:
        print(f"These top {top_n} contributors explain {explained_share:.1f}% of the total {direction}.")

def display_risk_metrics(risk_metrics_df):
    """
    Display risk metrics table and explanatory text.
    
    risk_metrics_df: DataFrame with risk metrics (rows: metrics, columns: periods like '1Y', '3Y', '5Y')
    """
    print("\n" + "="*60)
    print("PORTFOLIO RISK METRICS")
    print("="*60)
    
    # Format the DataFrame for display
    display_df = risk_metrics_df.copy()
    
    # Format each metric row
    for metric in display_df.index:
        if metric == 'Volatility':
            display_df.loc[metric] = display_df.loc[metric].apply(
                lambda x: f"{x*100:.2f}%" if x is not None and not pd.isna(x) else "N/A"
            )
        elif metric == 'Sharpe Ratio':
            display_df.loc[metric] = display_df.loc[metric].apply(
                lambda x: f"{x:.2f}" if x is not None and not pd.isna(x) else "N/A"
            )
        elif metric == 'Beta':
            display_df.loc[metric] = display_df.loc[metric].apply(
                lambda x: f"{x:.2f}" if x is not None and not pd.isna(x) else "N/A"
            )
        elif metric == 'Max Drawdown':
            display_df.loc[metric] = display_df.loc[metric].apply(
                lambda x: f"{x*100:.2f}%" if x is not None and not pd.isna(x) else "N/A"
            )
    
    print(display_df.to_string())
    print()
    
    # Get values for explanations (use 1Y if available, otherwise first available period)
    period_for_explanation = None
    for period in ['1Y', '3Y', '5Y']:
        if period in display_df.columns:
            period_for_explanation = period
            break
    
    if period_for_explanation:
        sharpe_val = risk_metrics_df.loc['Sharpe Ratio', period_for_explanation]
        beta_val = risk_metrics_df.loc['Beta', period_for_explanation]
        years_label = period_for_explanation.replace('Y', ' year' if period_for_explanation == '1Y' else ' years')
        
        # Sharpe ratio explanation
        if sharpe_val is not None and not pd.isna(sharpe_val):
            sharpe_str = f"{sharpe_val:.2f}"
            if sharpe_val < 0:
                sharpe_explanation = (
                    f"Your Sharpe ratio of {sharpe_str} over the last {years_label} indicates that your portfolio "
                    f"has underperformed relative to its risk level. A negative Sharpe ratio means your returns "
                    f"were lower than a risk-free investment, and the portfolio took on risk without adequate reward."
                )
            elif sharpe_val < 1:
                sharpe_explanation = (
                    f"Your Sharpe ratio of {sharpe_str} over the last {years_label} suggests your portfolio "
                    f"has generated moderate risk-adjusted returns. Generally, a Sharpe ratio above 1.0 is considered "
                    f"good, above 2.0 is very good, and above 3.0 is excellent. Your current ratio indicates "
                    f"decent performance relative to the volatility you've experienced."
                )
            elif sharpe_val < 2:
                sharpe_explanation = (
                    f"Your Sharpe ratio of {sharpe_str} over the last {years_label} indicates strong risk-adjusted "
                    f"performance. This means your portfolio has generated good returns relative to the amount of "
                    f"volatility (price swings) you've experienced. A Sharpe ratio above 1.0 is generally considered "
                    f"good, and yours shows you're being well-compensated for the risk you're taking."
                )
            else:
                sharpe_explanation = (
                    f"Your Sharpe ratio of {sharpe_str} over the last {years_label} indicates excellent risk-adjusted "
                    f"performance. This means your portfolio has generated outstanding returns relative to the volatility "
                    f"you've experienced. A Sharpe ratio above 2.0 is considered very good, and yours demonstrates "
                    f"exceptional efficiency in generating returns per unit of risk."
                )
            print(f"Sharpe Ratio Explanation:")
            print(f"  {sharpe_explanation}")
            print()
        
        # Beta explanation
        if beta_val is not None and not pd.isna(beta_val):
            beta_str = f"{beta_val:.2f}"
            if beta_val < 0.5:
                beta_explanation = (
                    f"Your portfolio's beta of {beta_str} over the last {years_label} indicates it's much less volatile "
                    f"than the market (S&P 500). A beta below 0.5 means your portfolio typically moves less than half "
                    f"as much as the market - when the market goes up or down 10%, your portfolio typically moves "
                    f"less than 5%. This suggests a more defensive, lower-risk portfolio."
                )
            elif beta_val < 0.8:
                beta_explanation = (
                    f"Your portfolio's beta of {beta_str} over the last {years_label} indicates it's less volatile "
                    f"than the market (S&P 500). A beta between 0.5 and 0.8 means your portfolio typically moves "
                    f"less than the market - when the market goes up or down 10%, your portfolio typically moves "
                    f"between 5-8%. This suggests a more conservative, defensive portfolio."
                )
            elif beta_val < 1.2:
                beta_explanation = (
                    f"Your portfolio's beta of {beta_str} over the last {years_label} indicates it moves similarly "
                    f"to the market (S&P 500). A beta close to 1.0 means your portfolio's price movements generally "
                    f"track the overall market. When the market goes up or down 10%, your portfolio typically moves "
                    f"about the same amount. This suggests a well-diversified portfolio that mirrors market behavior."
                )
            else:
                beta_explanation = (
                    f"Your portfolio's beta of {beta_str} over the last {years_label} indicates it's more volatile "
                    f"than the market (S&P 500). A beta above 1.2 means your portfolio typically moves more than "
                    f"the market - when the market goes up or down 10%, your portfolio typically moves more than 12%. "
                    f"This suggests a more aggressive, growth-oriented portfolio that amplifies market movements."
                )
            print(f"Beta Explanation:")
            print(f"  {beta_explanation}")
            print()

