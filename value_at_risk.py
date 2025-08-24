import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    df = yf.download(stock, start_date, end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {stock} between {start_date} and {end_date}")

        # Pick the adjusted close (fallback to 'Close' if needed)
    target_col = 'Adj Close' if 'Adj Close' in df.columns.get_level_values(0) else 'Close'

    # Handle both single- and multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # level 0 = field, level 1 = ticker
        s = df[target_col][stock]
    else:
        s = df[target_col]

    # Ensure it's a Series with the correct name, then return as 1-col DataFrame
    s = pd.Series(s, name=stock)
    return s.to_frame()


# this is how we calculate the Value at Risk (VaR) tomorrow (n=1)
def calculate_var(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1 - c))
    return var


# this is how we calculate the Value at Risk (VaR) for any number of days (n) in the future
def calculate_var_n(position, c, mu, sigma, n):
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return var


if __name__ == '__main__':
    ticker = 'MSFT'
    start = datetime.datetime(2022, 8, 1)
    end = datetime.datetime(2025, 8, 1)

    stock_data = download_data(ticker, start, end)

    stock_data['returns'] = np.log(stock_data[ticker] / stock_data[ticker].shift(1))
    stock_data = stock_data[1:]

    print(stock_data)

    # this is the investment (stocks or whatever else)
    S = 1e4

    # confidence level - this time it is 95%
    c = 0.95

    # number of days (n)
    n = 7

    # we assume daily returns are normally distributed
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    print('Value at risk is: $%0.2f' % calculate_var(S, c, mu, sigma))

    print('Value at risk over ' + str(n) + ' days is: $%0.2f' % calculate_var_n(S, c, mu, sigma, n))
