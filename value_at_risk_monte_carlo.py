import numpy as np
import pandas as pd
import datetime
import yfinance as yf

from markowitz_model import start_date


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


class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):

        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])
        # equation for the S(t) stock price
        # the random walk of our initial investment
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +
                                      self.sigma * np.sqrt(self.n) * rand)

        # we have to sort the stock prices to determine the percentile
        stock_price = np.sort(stock_price)

        # it depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


if __name__ == '__main__':

    S = 1e6   # this is the investment (stock or whatever)
    c = 0.95  # confidence level: this time it is 95%
    n = 1     # 1 day
    iterations = 100000 # number of paths in the Monte-Carlo simulation

    # historical data to approximate mean and standard deviation
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2025, 8, 1)

    ticker = 'C'

    # download stock data from Yahoo Finance
    data = download_data(ticker, start_date, end_date)

    # we can use pct_change() to calculate daily returns
    data['returns'] = data[ticker].pct_change()

    # we can assume daily returns to be normally distributed:
    # mean and variance (standard deviation)
    # can describe the process
    mu = np.mean(data['returns'])
    sigma = np.std(data['returns'])

    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
    print('Value at risk with Monte-Carlo simulation: $%.2f' % model.simulation())
