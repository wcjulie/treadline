# treadline

1. Finding Pivot Points
We'll start by identifying the pivot points (local highs and lows) in the price chart. This will be done using a function that checks for local minima and maxima in a moving window.

2. Detecting Support and Resistance Trend Lines
We'll detect support and resistance lines by iterating over all possible combinations of two pivot points (local lows for support and local highs for resistance) and check if the closing prices are always above (for support) or below (for resistance) the lines connecting these pivot points.

3. Removing Overlapping Trend Lines
We'll remove overlapping trend lines by comparing them based on their length and the number of price touches.

4. Plotting the Trend Lines
We'll plot the detected trend lines on the candlestick chart using matplotlib and mplfinance libraries. Support trend lines will be colored blue and resistance trend lines will be colored black.
