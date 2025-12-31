import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def plot_cumulative_returns(portfolio_index, benchmark_index, benchmark_label='SPY'):
    # Align on common dates
    common_index = portfolio_index.index.intersection(benchmark_index.index)
    port = portfolio_index.loc[common_index]
    bench = benchmark_index.loc[common_index]

    plt.figure(figsize=(10, 6))
    plt.plot(port.index, port.values, label='Portfolio')
    plt.plot(bench.index, bench.values, label=benchmark_label)

    plt.xlabel('Date')
    plt.ylabel('Growth of $100')
    plt.title("Cumulative Returns: Portfolio vs. Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_return_contributions(contributions, top_n=5, period_label="YTD", weights=None):
    """
    Plot top N positive and top N negative contributors as a horizontal bar chart.
    
    Filters out cash tickers from visualization but includes them in totals.
    Adds note if cash weight > 5%.

    contributions: pandas.Series indexed by ticker, values = contribution in decimal
                   (e.g. 0.025 = +2.5 percentage points)
    weights: optional dict/Series of weights to detect cash position
    """
    # Import cash detection helper
    from portfolio_tool.analytics import is_cash_ticker, get_cash_weight
    
    # Drop NaNs
    contributions = contributions.dropna()

    if contributions.empty:
        print("No contributions to plot.")
        return

    # Filter out cash tickers from visualization
    non_cash_contributions = contributions[[ticker for ticker in contributions.index 
                                           if not is_cash_ticker(ticker)]]
    
    if non_cash_contributions.empty:
        print("No non-cash contributions to plot.")
        return

    # Separate positive and negative contributors (excluding cash)
    positives = non_cash_contributions[non_cash_contributions > 0].sort_values(ascending=False).head(top_n)
    negatives = non_cash_contributions[non_cash_contributions < 0].sort_values(ascending=True).head(top_n)

    # Combine: negatives first (top), positives later (bottom)
    combined = pd.concat([negatives, positives])

    # Convert to percentage points
    values = combined * 100.0
    tickers = combined.index

    # Choose colors: red for negative, green for positive
    colors = ["tab:red" if v < 0 else "tab:green" for v in values]

    plt.figure(figsize=(10, 5))
    plt.axvline(0, linewidth=1, color="black")  # vertical line at 0 for reference
    bars = plt.barh(tickers, values, color=colors)

    # Add labels at the end of each bar
    for bar, v in zip(bars, values):
        xpos = bar.get_width()
        sign = "+" if v >= 0 else ""
        plt.text(
            xpos + (0.1 if v >= 0 else -0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{v:.2f}%",
            va="center",
            ha="left" if v >= 0 else "right",
        )

    # Title logic depending on whether we have negatives
    if len(negatives) > 0 and len(positives) > 0:
        title = f"Top {top_n} Positive and Negative Return Contributions ({period_label})"
    elif len(positives) > 0:
        title = f"Top {top_n} Positive Return Contributions ({period_label})"
    else:
        title = f"Top {top_n} Negative Return Contributions ({period_label})"

    plt.xlabel("Return Contribution (percentage points)")
    plt.title(title)
    
    # Add note if cash weight > 5%
    if weights is not None:
        cash_weight = get_cash_weight(weights)
        if cash_weight > 0.05:  # 5%
            note_text = (f"Note: A cash position ({cash_weight*100:.1f}%) reduces upside and downside; "
                        f"cash is included in totals but not shown as a bar.")
            plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=9, style="italic", wrap=True)

    plt.tight_layout()
    plt.show()

def plot_risk_contributions(contributions, top_n=5, period_label="YTD", weights=None):
    """
    Plot top N risk contributors as a horizontal bar chart.
    
    Filters out cash tickers from visualization and renormalizes to sum to 100%.
    Adds note if cash weight > 5%.

    contributions: pandas.Series indexed by ticker, values = risk contribution as decimal
                   (e.g. 0.35 = 35% of total portfolio risk)
    weights: optional dict/Series of weights to detect cash position
    """
    # Import cash detection helper
    from portfolio_tool.analytics import is_cash_ticker, get_cash_weight
    
    # Drop NaNs
    contributions = contributions.dropna()

    if contributions.empty:
        print("No risk contributions to plot.")
        return

    # Filter out cash tickers from visualization
    non_cash_contributions = contributions[[ticker for ticker in contributions.index 
                                         if not is_cash_ticker(ticker)]]
    
    if non_cash_contributions.empty:
        print("No non-cash risk contributions to plot.")
        return
    
    # Renormalize to sum to 100% across non-cash holdings
    total_non_cash = non_cash_contributions.sum()
    if total_non_cash > 0:
        non_cash_contributions = non_cash_contributions / total_non_cash
    else:
        print("No meaningful risk contributions to plot.")
        return

    # Get top N contributors (all should be positive for risk)
    top_contributors = non_cash_contributions.sort_values(ascending=False).head(top_n)

    # Convert to percentage
    values = top_contributors * 100.0
    tickers = top_contributors.index

    # Use blue color for risk contributions
    colors = ["tab:blue"] * len(values)

    plt.figure(figsize=(10, 5))
    bars = plt.barh(tickers, values, color=colors)

    # Add labels at the end of each bar
    for bar, v in zip(bars, values):
        xpos = bar.get_width()
        plt.text(
            xpos + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%",
            va="center",
            ha="left",
        )

    title = f"Top {top_n} Risk Contributors ({period_label})"
    plt.xlabel("Risk Contribution (%)")
    plt.title(title)
    
    # Add note if cash weight > 5%
    if weights is not None:
        cash_weight = get_cash_weight(weights)
        if cash_weight > 0.05:  # 5%
            note_text = (f"Cash lowers portfolio volatility but is excluded from the risk "
                        f"contribution bars for readability.")
            plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=9, style="italic", wrap=True)

    plt.tight_layout()
    plt.show()

def plot_monthly_returns_heatmap(monthly_returns, title = "Monthly Portfolio Returns"):
    """
    Plot a heatmap of monthly returns.

    """
    # convert to percentage
    data = monthly_returns * 100.0
    years = data.index.astype(int).tolist()
    months = data.columns.tolist()

    # prepare matrix, handling NaNs
    mat = data.values
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    max_abs = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', norm=norm)

    # set ticks
    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(months)
    ax.set_yticklabels(years)

    # rotate month labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate each cell with value
    for i in range(len(years)):
        for j in range(len(months)):
            value = mat[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j, i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                )
            
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    fig.colorbar(im, ax=ax, label="Monthly Return (%)")

    plt.tight_layout()
    plt.show()

def plot_sector_allocation(sector_weights, title="Portfolio Allocation by Sector"):
    """
    Plot a pie chart showing portfolio allocation by sector.
    
    sector_weights: pandas.Series indexed by sector, values = weights (0-1)
    title: chart title
    """
    if sector_weights.empty:
        print("No sector data available to plot.")
        return
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate colors using a colormap
    colors = plt.cm.Set3(range(len(sector_weights)))
    
    # Create pie chart with labels showing percentage
    wedges, texts, autotexts = ax.pie(
        sector_weights.values,
        labels=sector_weights.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    
    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(correlation_matrix, title="Portfolio Asset Correlation Matrix"):
    """
    Plot a heatmap of the correlation matrix between portfolio assets.
    
    correlation_matrix: DataFrame with correlation values (tickers x tickers)
    title: chart title
    
    Color coding: Lower correlation (better diversification) = cool colors (blue/green)
                   Higher correlation (worse diversification) = warm colors (red/orange)
    """
    if correlation_matrix.empty:
        print("No correlation data available to plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data - ensure symmetric and handle NaN
    corr_data = correlation_matrix.values
    tickers = correlation_matrix.index.tolist()
    
    # Use a reversed colormap where:
    # - Low correlation (0 to ~0.5) = cool colors (blue/cyan) = better
    # - High correlation (~0.5 to 1) = warm colors (yellow/red) = worse
    # Using 'RdYlGn_r' reversed or 'coolwarm' - let's use 'RdYlGn_r' for better contrast
    # Actually, let's use a custom approach: 'coolwarm' but reversed
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed (red=high corr, green=low corr)
    
    # Create heatmap
    im = ax.imshow(corr_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right')
    ax.set_yticklabels(tickers)
    
    # Add text annotations with correlation values
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            value = corr_data[i, j]
            # Skip diagonal (always 1.0) or show it differently
            if i == j:
                text = ax.text(j, i, '1.00', ha="center", va="center",
                             color="black", fontweight='bold', fontsize=9)
            else:
                # Use white text on dark backgrounds, black on light
                text_color = 'white' if abs(value) > 0.5 else 'black'
                text = ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                             color=text_color, fontsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Correlation')
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Assets', fontsize=12)
    ax.set_ylabel('Assets', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_risk_return_scatter(risk_return_data, title="Risk-Return Analysis (5 Years)"):
    """
    Plot a risk-return scatter plot showing annualized return vs volatility.
    
    risk_return_data: dict with keys as labels and values as (volatility, return) tuples
                     e.g., {'Portfolio': (0.15, 0.12), 'SPY (S&P 500)': (0.18, 0.14)}
    title: chart title
    """
    if not risk_return_data:
        print("No risk-return data available to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Map labels to legend descriptions
    legend_labels = {
        'Portfolio': 'Portfolio',
        'SPY (S&P 500)': 'SPY (S&P 500)',
        'QQQ (Nasdaq-100)': 'QQQ (Nasdaq-100)',
        'AGG (Investment-Grade Bonds)': 'AGG (Investment-Grade Bonds)',
        'ACWI (All Country World Index)': 'ACWI (All Country World Index)'
    }
    
    # Plot each point with simple, clean styling
    for label, (vol, ret) in risk_return_data.items():
        if vol is None or ret is None:
            continue
        
        legend_label = legend_labels.get(label, label)
        plt.scatter(vol * 100, ret * 100, label=legend_label, s=100)
    
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier(ef_data, title="Efficient Frontier Analysis"):
    """
    Plot the efficient frontier with all points and Capital Market Line.
    
    ef_data: dict returned from compute_efficient_frontier_analysis()
    title: chart title
    """
    if ef_data is None:
        print("No efficient frontier data available to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot random portfolios (faint scatter in background)
    if 'random_portfolios' in ef_data and not ef_data['random_portfolios'].empty:
        random_df = ef_data['random_portfolios']
        plt.scatter(random_df['vol'] * 100, random_df['ret'] * 100,
                   alpha=0.1, s=10, color='gray', label='Random Portfolios')
    
    # Plot efficient frontier (solid line) - already sorted by vol
    if 'frontier' in ef_data and not ef_data['frontier'].empty:
        frontier_df = ef_data['frontier'].copy()
        # Ensure sorted by volatility for smooth line (should already be sorted, but double-check)
        frontier_df = frontier_df.sort_values('vol').reset_index(drop=True)
        # Only plot if we have at least 2 points
        if len(frontier_df) >= 2:
            plt.plot(frontier_df['vol'] * 100, frontier_df['ret'] * 100,
                    'b-', linewidth=2, label='Efficient Frontier', zorder=2)
    
    # Plot individual assets (labeled scatter points)
    if 'asset_points' in ef_data and not ef_data['asset_points'].empty:
        asset_df = ef_data['asset_points']
        for _, row in asset_df.iterrows():
            plt.scatter(row['vol'] * 100, row['ret'] * 100,
                       s=100, marker='o', color='blue', alpha=0.6, zorder=3)
            plt.annotate(row['ticker'], 
                        xy=(row['vol'] * 100, row['ret'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    # Plot user portfolio (distinct labeled point)
    if 'portfolio_point' in ef_data:
        port_point = ef_data['portfolio_point']
        plt.scatter(port_point['vol'] * 100, port_point['ret'] * 100,
                   s=150, marker='D', color='red', edgecolors='black', linewidth=2,
                   label='Your Portfolio', zorder=4)
        plt.annotate('Your Portfolio',
                    xy=(port_point['vol'] * 100, port_point['ret'] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    # Plot benchmarks (distinct labeled points)
    if 'benchmark_points' in ef_data and not ef_data['benchmark_points'].empty:
        bench_df = ef_data['benchmark_points']
        for _, row in bench_df.iterrows():
            plt.scatter(row['vol'] * 100, row['ret'] * 100,
                       s=120, marker='s', color='green', edgecolors='black', linewidth=1.5,
                       zorder=3)
            plt.annotate(row['ticker'],
                        xy=(row['vol'] * 100, row['ret'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    # Plot tangency portfolio (star marker)
    if 'tangency' in ef_data and ef_data['tangency'] is not None:
        tan_point = ef_data['tangency']
        plt.scatter(tan_point['vol'] * 100, tan_point['ret'] * 100,
                   s=200, marker='*', color='gold', edgecolors='black', linewidth=2,
                   label='Tangency Portfolio', zorder=5)
        plt.annotate('Tangency',
                    xy=(tan_point['vol'] * 100, tan_point['ret'] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    # Plot Capital Market Line (CML)
    if 'tangency' in ef_data and ef_data['tangency'] is not None and 'risk_free_rate' in ef_data:
        tan_point = ef_data['tangency']
        rf = ef_data['risk_free_rate']
        
        # CML: line from (0, rf) to (sigma_tan, mu_tan)
        x_cml = [0, tan_point['vol'] * 100]
        y_cml = [rf * 100, tan_point['ret'] * 100]
        plt.plot(x_cml, y_cml, 'r--', linewidth=1.5, alpha=0.7, label='Capital Market Line')
    
    plt.xlabel('Volatility (σ) %')
    plt.ylabel('Expected Return (μ) %')
    plt.title(title)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

