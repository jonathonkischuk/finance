import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.Ticker(stock)
    data['Price'] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(data)


def calculate_returns(stock_data):
    stock_data['Price'] = np.log(stock_data['Price'] / stock_data['Price'].shift(1))
    return stock_data[1:]


def show(stock_data):
    plt.hist(stock_data, bins=700)
    stock_variance = stock_data.var()
    stock_mean = stock_data.mean()
    sigma = np.sqrt(stock_variance)     # This is the Standard Deviation
    x = np.linspace(stock_mean - 5 * sigma, stock_mean + 5 * sigma, 100)
    plt.plot(x, norm.pdf(x, stock_mean, sigma))
    plt.show()


if __name__ == '__main__':
    stock_data = download_data('MSFT', '2015-01-01', '2025-07-31')
    show(calculate_returns(stock_data))
