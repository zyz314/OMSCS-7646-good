import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from util import get_data
from marketsimcode import compute_portvals

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    """
    Code implementing a TheoreticallyOptimalStrategy. It should implement testPolicy(), which returns a trades Pandas.DataFrame

    Parameters
        symbol    - the stock symbol to act on
        sd        - A DateTime object that represents the start date
        ed        - A DateTime object that represents the end date
        sv        - Start value of the portfolio

    Returns
        A single column data frame, indexed by date, whose values represent trades for each trading day
       (from the start date to the end date of a given period). Legal values are +1000.0 indicating a BUY
       of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000
       and -2000 for trades are also legal so long as net holdings are constrained to -1000, 0, and 1000.
       Note: The format of this data frame differs from the one developed in a prior project.

    Return Type
        pandas.DataFrame
    """
    dates = pd.date_range(start=sd, end=ed)
    trades = create_trades(dates=dates, symbol=symbol)
    
    port_vals = compute_portvals(trades=trades, start_val=sv, commission=0, impact=0)
    port_vals = port_vals/port_vals.iloc[0]
    
    benchmark = create_benchmark(dates=dates,symbol=symbol)
    benchmark_vals = compute_portvals(trades=benchmark, start_val=sv, commission=0, impact=0)
    benchmark_vals = benchmark_vals/benchmark_vals.iloc[0]
    
    create_chart(port_vals,benchmark_vals,symbol)
    create_table(port_vals,benchmark_vals)
    
    return trades


def author():
    """
    Returns
        The GT username of the student

    Return type
        str
    """
    return "mshihab6"

def create_prices(dates, symbols):
    prices = get_data(dates=dates, symbols=symbols)
    prices = prices.dropna(subset=["SPY"])
    prices = prices[symbols]
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    prices["Cash"] = 1.0
    return prices

def create_trades(dates,symbol):
    symbols = [symbol]
    
    prices = create_prices(dates=dates,symbols=symbols)
    
    trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i, (date,values) in enumerate(trades.iloc[:-1].iterrows()):
        total_shares = trades.loc[:date,symbols].sum()[0]
        is_tomorrow_more = (prices.iloc[i+1]>prices.iloc[i])[symbol]
        if is_tomorrow_more: 
            action = buy_action(total_shares)
        else:
            action = sell_action(total_shares)
        trades.loc[date,symbols]=action
        trades.loc[date,"Cash"]=-action*prices.loc[date,symbol]
    return trades

def create_benchmark(dates,symbol):
    symbols = [symbol]
    
    prices = create_prices(dates=dates,symbols=symbols)
    
    benchmark = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    benchmark.iloc[0][symbols] = 1000
    benchmark.iloc[0]["Cash"] = prices.iloc[0][symbols]*-1000
    return benchmark

def create_chart(port_vals,benchmark_vals,symbol):
    fig, ax = plt.subplots()
    benchmark_vals.plot(ax = ax, color = "purple", label="Benchmark Strategy")
    port_vals.plot(ax = ax, color = "red", label="Theoretically Optimal Strategy")
    ax.grid(visible=True,linestyle=':')
    ax.legend()
    ax.set_title(label=f"TheoreticallyOptimalStrategy vs Benchmark on {symbol} symbol")
    ax.set_xlabel(xlabel="Date")
    ax.set_ylabel(ylabel="Normalized Value")
    plt.savefig("p6_tos_chart.png")
    
def create_table(port_vals,bechmark_vals):
    port_vals_data = [
        get_cr(port_vals),
        get_adr(port_vals),
        get_sddr(port_vals),
        get_sr(port_vals)
    ]
    benchmark_data = [
        get_cr(bechmark_vals),
        get_adr(bechmark_vals),
        get_sddr(bechmark_vals),
        get_sr(bechmark_vals)
    ]
    port_stats = pd.DataFrame(data={"Portfolio":port_vals_data,
                                    "Benchmark":benchmark_data},
                              index=["cum_ret","avg_dr","ssdr","sharpe_ratio"])
    port_stats = np.round(a=port_stats,decimals=6)
    port_stats.to_csv("p6_results.csv")
    with open("p6_results.html", "w") as f:
        f.writelines(f"<html>")
        f.writelines(f"<h1>p6_results for TheoreticallyOptimalStrategy.py produced by {author()}: 903687444</h1>")
        f.writelines("<h2>Table 1: TOS Stats </h2>")
        f.writelines(port_stats.to_html())

def buy_action(share_count:int):
    return 1000-share_count
def sell_action(share_count:int):
    return -1000 - share_count

def get_cr(port_val):
    """
    This function will return the cumulative return of a profile
    According to the lectures:
    Cumulative Return = (port_val[-1]/port_val[0]) - 1

    :param port_val: A pandas DataFrame object that contains the portfolio value
    :type port_val: pd.DataFrame
    :return: A pandas DataFrame object that represents the cumulative return values of the portfolio
    :rtype: pd.DataFrame
    """
    cr = (port_val.iloc[-1] / port_val.iloc[0]) - 1
    return cr


def get_daily_rets(port_val):
    """
    This function will return the average daily return of a profile
    According to the lectures:
    daily_rets = (df[1:]/df[:-1].values)-1
    Or in other words, today's value / yesterday's value

    :param port_val: A pandas DataFrame object that contains the portfolio value
    :type port_val: pd.DataFrame
    :return: A pandas DataFrame object that represents the daily return values of the portfolio
    :rtype: pd.DataFrame
    """
    daily_rets = (port_val / port_val.shift(1)) - 1
    return daily_rets


def get_adr(port_val):
    """
    This function will return the average daily return of a profile
    According to the lectures:
    Average Daily Return = daily_rets.mean()

    :param port_val: A pandas DataFrame object that contains the portfolio value
    :type port_val: pd.DataFrame
    :return: A float that is equal to the average values of the portfolio
    :rtype: float
    """
    daily_rets = get_daily_rets(port_val)
    adr = daily_rets.mean()
    return adr


def get_sddr(port_val):
    """
    This function will return the Standard Deviation of Daily Return of a profile
    According to the lectures:
    Standard Deviation of Daily Return = daily_rets.std()
    Note: We need sample standard deviation (Thank you Andrew Rife)

    :param port_val: A pandas DataFrame object that contains the portfolio value
    :type port_val: pd.DataFrame
    :return: A float that is equal to the standard deviation values of the portfolio
    :rtype: float
    """
    daily_rets = get_daily_rets(port_val)
    sddr = np.std(daily_rets,ddof=1)
    return sddr


def get_sr(port_val, risk_free_rate = 0):
    """
    This function will return the Sharpe Ratio of a profile

    :param port_val: A pandas DataFrame object that contains the portfolio value
    :type port_val: pd.DataFrame
    :param risk_free_rate: A float that represents the risk-free rate used in the Sharpe Ratio Calculation
    :type risk_free_rate: float
    :return: The Sharpe Ratio of the portfolio provided
    :rtype: float
    """
    sr = ((get_adr(port_val)-risk_free_rate) / get_sddr(port_val))*math.sqrt(252)
    return sr