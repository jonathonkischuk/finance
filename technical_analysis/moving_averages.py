from cProfile import label

import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.Ticker(stock)
    data['Price'] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(data)


def construct_signals(data, short_period, long_period):
    data['Short SMA'] = data['Price'].ewm(span=short_period, adjust=False).mean()
    data['Long SMA'] = data['Price'].ewm(span=long_period, adjust=False).mean()
    data = data.dropna()
    print(data)


def plot_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Price'], label='Stock Price', color='black')
    plt.plot(data['Short SMA'], label='Short SMA', color='red')
    plt.plot(data['Long SMA'], label='Long SMA', color='blue')
    plt.title("Moving Average (MA) Indicators")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()


if __name__ == "__main__":
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2025, 7, 31)
    stock_data = download_data('IBM', start, end)
    construct_signals(stock_data, short_period=30, long_period=200)
    stock_data = stock_data.dropna()
    plot_data(stock_data)
