import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import yfinance as yf
import plotly.io as pio


def pivot_low(data, left, right):
    pivot = np.full(data.shape, np.nan)
    for i in range(left, len(data) - right):
        if all(data[i] <= data[i - left:i]) and all(data[i] <= data[i + 1:i + right + 1]):
            pivot[i] = data[i]
    return pivot

def pivot_high(data, left, right):
    pivot = np.full(data.shape, np.nan)
    for i in range(left, len(data) - right):
        if all(data[i] >= data[i - left:i]) and all(data[i] >= data[i + 1:i + right + 1]):
            pivot[i] = data[i]
    return pivot

def count_touches(x_values, line_values, price_values, threshold=0.01):
    distances = np.abs(price_values - line_values)
    touch_mask = distances <= threshold * price_values
    touches = np.sum(touch_mask)
    return touches

from itertools import combinations


def detect_trendlines(data, pivot_points, is_support=True, min_pivots=5):
    trendlines = []
    price_col = 'Low' if is_support else 'High'
    compare_price = 'Close'

    for combo in combinations(pivot_points.index, min_pivots):
        x_values = [data.index.get_loc(idx) for idx in combo]
        y_values = [pivot_points.loc[idx, price_col] for idx in combo]

        x1, y1 = x_values[0], y_values[0]
        x2, y2 = x_values[-1], y_values[-1]

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        line_values = slope * np.arange(x1, x2 + 1) + intercept

        if is_support:
            valid = np.all(data.iloc[x1:x2 + 1][compare_price].values >= line_values)
        else:
            valid = np.all(data.iloc[x1:x2 + 1][compare_price].values <= line_values)

        if valid:
            touches = count_touches(np.arange(x1, x2 + 1), line_values, data.iloc[x1:x2 + 1][price_col].values)
            trendlines.append((data.index[x1], y1, data.index[x2], y2, touches, x2 - x1))

    return trendlines

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:  # Lines are parallel
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        return True
    return False

def remove_overlapping_trendlines(trendlines, threshold=0.1):
    sorted_trendlines = sorted(trendlines, key=lambda t: (t[4] * np.log((t[2] - t[0]).days)), reverse=True)
    final_trendlines = []

    for trendline in sorted_trendlines:
        x1, y1, x2, y2, touches, _ = trendline
        overlapping = False
        for existing in final_trendlines:
            ex1, ey1, ex2, ey2, _, _ = existing

            # Convert datetime to ordinal for calculation
            x1_ord, x2_ord = x1.toordinal(), x2.toordinal()
            ex1_ord, ex2_ord = ex1.toordinal(), ex2.toordinal()

            # Check for overlap in x-axis
            if (ex1_ord <= x1_ord <= ex2_ord or ex1_ord <= x2_ord <= ex2_ord or
                    x1_ord <= ex1_ord <= x2_ord or x1_ord <= ex2_ord <= x2_ord):
                # Check if lines are close to each other
                if ex2_ord != ex1_ord:  # Avoid division by zero
                    y1_on_existing = ey1 + (ey2 - ey1) * (x1_ord - ex1_ord) / (ex2_ord - ex1_ord)
                    y2_on_existing = ey1 + (ey2 - ey1) * (x2_ord - ex1_ord) / (ex2_ord - ex1_ord)
                    if (abs(y1 - y1_on_existing) / y1 < threshold and
                            abs(y2 - y2_on_existing) / y2 < threshold):
                        overlapping = True
                        break

                # Check for intersection
                if line_intersection((x1_ord, y1, x2_ord, y2), (ex1_ord, ey1, ex2_ord, ey2)):
                    overlapping = True
                    break

        if not overlapping:
            final_trendlines.append(trendline)

    return final_trendlines

def analyze_stock(ticker, start_date, end_date, pivot_len):
    # Download data using yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    # Parameters
    min_pivots = 3  # Minimum number of consecutive pivots to form a trendline

    # Calculate pivot points
    df['PivotLow'] = pivot_low(df['Low'].values, pivot_len, pivot_len)
    df['PivotHigh'] = pivot_high(df['High'].values, pivot_len, pivot_len)

    # Detect support and resistance trendlines
    support_pivots = df[~np.isnan(df['PivotLow'])]
    resistance_pivots = df[~np.isnan(df['PivotHigh'])]

    support_lines = detect_trendlines(df, support_pivots, is_support=True, min_pivots=min_pivots)
    resistance_lines = detect_trendlines(df, resistance_pivots, is_support=False, min_pivots=min_pivots)

    # Remove overlapping trendlines
    final_support_lines = remove_overlapping_trendlines(support_lines)
    final_resistance_lines = remove_overlapping_trendlines(resistance_lines)

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])

    # Add support lines to the plot
    for x1, y1, x2, y2, touches, _ in final_support_lines:
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(color="blue", width=3),
            name=f"Support ({touches} touches)"
        ))

    # Add resistance lines to the plot
    for x1, y1, x2, y2, touches, _ in final_resistance_lines:
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(color="black", width=3),
            name=f"Resistance ({touches} touches)"
        ))

    # Add markers for pivot lows
    fig.add_trace(go.Scatter(
        x=df.index[~np.isnan(df['PivotLow'])],
        y=df['PivotLow'][~np.isnan(df['PivotLow'])],
        mode='markers',
        marker=dict(color='blue', symbol='triangle-down', size=8),
        name='Pivot Low'
    ))

    # Add markers for pivot highs
    fig.add_trace(go.Scatter(
        x=df.index[~np.isnan(df['PivotHigh'])],
        y=df['PivotHigh'][~np.isnan(df['PivotHigh'])],
        mode='markers',
        marker=dict(color='red', symbol='triangle-up', size=8),
        name='Pivot High'
    ))

    # Update the layout
    fig.update_layout(
        title=f'{ticker} Daily Stock Candlestick Chart with Support and Resistance Trendlines (Pivot Length: {pivot_len})',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )

    # Show the plot
    fig.show()

    # Save the plot to an HTML file
    pio.write_html(fig, file=f"{ticker}_candlestick_with_trendlines.html", auto_open=True)

    # Print the number of detected trendlines
    print(f"Number of support trendlines: {len(final_support_lines)}")
    print(f"Number of resistance trendlines: {len(final_resistance_lines)}")

# Example usage with single-line input
# input_str = input("Enter ticker, pivot length (number of periods determine the local extremes), start date, end date (e.g., TSLA 20 2022-01-01 2023-01-01): ")
# ticker, pivot_len, start_date, end_date  = input_str.split()
# pivot_len = int(pivot_len)
# analyze_stock(ticker, start_date, end_date, pivot_len)

analyze_stock('SPY', '2020-01-01', '2024-01-01',20)
analyze_stock('TSLA', '2020-01-01', '2024-01-01',20)
analyze_stock('TXN', '2020-01-01', '2024-01-01',20)
analyze_stock('AAPL', '2020-01-01', '2024-01-01',20)