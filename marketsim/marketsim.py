""""""
"""MC2-P1: Market simulator.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Mahmoud Shihab (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: mshihab6 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903687444 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""

import datetime as dt
import os

import numpy as np
import pandas as pd
import math
from util import get_data, plot_data


def compute_portvals(
        orders_file="./orders/orders.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """  		  	   		 	   			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	   			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	   			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
    :type start_val: int  		  	   		 	   			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
    """
    # this is the function the autograder will call to test your code  		  	   		 	   			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	   			  		 			     			  	 
    # TODO: Your code here  		  	   		 	   			  		 			     			  	 

    # In the template, instead of computing the value of the portfolio, we just  		  	   		 	   			  		 			     			  	 
    # read in the value of IBM over 6 months  		  	   		 	   			  		 			     			  	 
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    orders = pd.read_csv(orders_file, parse_dates=True, na_values=['nan'])
    orders.Date = pd.to_datetime(orders.Date)
    orders = orders.sort_values(by="Date", ascending=True)

    # Get required data
    symbols = orders.Symbol.unique()
    start_date = orders.Date.min()
    end_date = orders.Date.max()
    dates = pd.date_range(start_date, end_date)

    # From orders, create Prices
    prices = get_data(symbols, dates) # Need SPY to drop non-trading days
    prices = prices.dropna(subset=["SPY"])[symbols] # Just to be safe, drop nan values in SPY and only take required symboles
    prices = prices.fillna(method="ffill").fillna(method="bfill") # Just to be safe, fill forward and backwards
    prices["Cash"] = 1.0  # NEEDS TO BE A FLOATING VALUE. WILL INTRODUCE ROUNDING ERRORS IF NOT!!!

    # Copy Prices, Call it Trades
    trades = create_trades(orders, prices, commission, impact)

    # Create Holdings
    holdings = create_holdings(start_val, trades)

    # Create Values
    values = prices * holdings

    # Add columns
    port_vals = values.sum(axis=1)

    return port_vals

    # return rv
    # return portvals


def create_trades(orders, prices, commission, impact):
    trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # 0.0 since the columns are meant to be used in holdings, and they need to be floats
    # If not floats, they will introduce rounding errors
    for _, r in orders.iterrows():
        date = r.Date
        #check to make sure that the order date is a date found in prices
        if date not in prices.index: continue
        sym = r.Symbol
        shares = r.Shares
        value = prices.loc[date][sym]

        sym_multiplier = -1 if r.Order == "SELL" else 1
        # We subtract shares when we sell and add orders when we buy
        trades.loc[date][sym] += (shares * sym_multiplier)

        cash = value * shares
        deductions = (cash * impact) + commission

        cash_multiplier = -1 if r.Order == "BUY" else 1
        # We add cash when we sell and remove cash when we buy
        trades.loc[date]["Cash"] += (cash_multiplier * cash) - deductions
    return trades


def create_holdings(start_val, trades):
    # https://stackoverflow.com/questions/73968466/in-python-is-there-a-way-to-vectorize-adding-the-previous-row-to-the-current-on
    # holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    # holdings.iloc[0].Cash = start_val
    # holdings += trades
    # for i in range(1, len(holdings)):
    #     holdings.iloc[i] += holdings.iloc[i - 1]
    holdings = trades.copy()
    holdings.iloc[0].Cash += start_val
    holdings = holdings.cumsum()
    return holdings


def author():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """
    return "mshihab6"  # Change this to your user ID


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


def test_code():
    """  		  	   		 	   			  		 			     			  	 
    Helper function to test code  		  	   		 	   			  		 			     			  	 
    """
    # this is a helper function you can use to test your code  		  	   		 	   			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	   			  		 			     			  	 
    # Define input parameters  		  	   		 	   			  		 			     			  	 

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders  		  	   		 	   			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	   			  		 			     			  	 
    else:
        print("warning, code did not return a DataFrame")

        # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	   			  		 			     			  	 
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    normalized_portvals = portvals / portvals.iloc[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        get_cr(normalized_portvals),
        get_adr(normalized_portvals),
        get_sddr(normalized_portvals),
        get_sr(normalized_portvals)
    ]

    prices = get_data(["$SPX"], pd.date_range(start_date, end_date))

    prices_SPX = prices["$SPX"]
    normalized_SPX = prices_SPX / prices_SPX.iloc[0]
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = [
        get_cr(normalized_SPX),
        get_adr(normalized_SPX),
        get_sddr(normalized_SPX),
        get_sr(normalized_SPX)
    ]

    prices_SPY = prices["SPY"]
    normalized_SPY = prices_SPY / prices_SPY.iloc[0]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        get_cr(normalized_SPY),
        get_adr(normalized_SPY),
        get_sddr(normalized_SPY),
        get_sr(normalized_SPY)
    ]

    # Compare portfolio against $SPX  		  	   		 	   			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of $SPX: {sharpe_ratio_SPX}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of $SPX: {cum_ret_SPX}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of $SPX: {std_daily_ret_SPX}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of $SPX: {avg_daily_ret_SPX}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
