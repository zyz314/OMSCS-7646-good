""""""
"""MC1-P2: Optimize a portfolio.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo
import math


# Helper Functions
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
    cr = (port_val[-1] / port_val[0]) - 1
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


def get_price_dfs(allocs, prices, starting_investment=1):
    """
    This function will calculate all the required pricing dataframes given the starting investment and the allocations

    :param allocs: A numpy array that represents the allocations needed for all the price dataframe calculations
    :type allocs: np.ndarray
    :param prices: A pandas DataFrame object that contains the prices for all the provided ticker symbols of the portfolio
    :type prices: pd.DataFrame
    :param starting_investment: A float that represents the currency amount of starting investment
    :type starting_investment: float
    :return: A pandas DataFrames object that represents
    - Normalized Prices
    - Allocated Prices
    - Portfolio Value with Starting Investment Added per Stock Ticker
    - Portfolio Values
    :rtype: tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame)
    """
    # Step 1: Normalizing the Prices
    normalized_prices = prices / prices.iloc[0]
    # Step 2: Multiply the allocations to the normalized prices, so we can start the process of getting pos_val
    allocated_prices = normalized_prices * allocs
    # Step 3: Multiply by initial investment (Note: Since we weren't given one, I'll set it to 1)
    pos_val = allocated_prices * starting_investment
    # Step 4: Getting port_val from allocated_prices 
    port_val = pos_val.sum(axis=1)  # axis = 1 means by row
    return normalized_prices, allocated_prices, pos_val, port_val


def optimization_function(allocs, prices):
    """
    This function that will be used in the minimizing function

    :param allocs: A numpy array that represents the allocations needed for all the price dataframe calculations
    :type allocs: np.ndarray
    :param prices: A pandas DataFrame object that contains the prices for all the provided ticker symbols of the portfolio
    :type prices: pd.DataFrame
    :return: The Sharpe Ratio of the portfolio provided that will be the metric that should be minimized
    :rtype: float
    """
    port_val = get_price_dfs(allocs, prices)[3]
    sr = get_sr(port_val)
    return -sr


def create_plot(df):
    """
    This function will save a plot needed for comparing the portoflio with SPY

    :param df: A pandas dataframe that contains the portfolio and SPY values
    :type df: pd.DataFrame
    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.hlines(y=1, xmin=df.index[0], xmax=df.index[-1], colors="grey", linestyles="dotted")
    ax.set_title("Daily Portfolio Value and SPY")
    ax.set_ylabel("Normalized Price")
    ax.set_xlabel("Date")
    plt.grid(visible=True, color='grey', linestyle='dotted')
    plt.tight_layout()
    plt.savefig("Figure1.png")


# This is the function that will be tested by the autograder  		  	   		 	   			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	   			  		 			     			  	 
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		 	   			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   			  		 			     			  	 
    statistics.  		  	   		 	   			  		 			     			  	 

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   			  		 			     			  	 
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  
    normalized_SPY = prices_SPY / prices_SPY.iloc[0] # Normalize SPY for plotting later

    # find the allocations for the optimal portfolio  		  	   		 	   			  		 			     			  	 
    # note that the values here ARE NOT meant to be correct for a test case  		

    # We are going st set up our initial guess  	   		 	   			  		 			     			  	 
    n = len(syms)
    init_allocs = np.asarray(
        [1 / n] * n  # Initial Guess
    )

    # As stated in the lesson on missing data, we need to forward fill and then backwards fill to remove any nulls
    # Thanks to Charlie Faber for his wonderful discussion on the topic
    prices = prices.ffill()  # Forward fill to ensure that gaps since the start of the stocks start are filled up
    prices = prices.bfill()  # Backwards fill *SECOND* since the gaps before the start of the stocks start to fill up
    # Order here is important. If you backwards fill first, you will only forward fill the price starting at the last
    # null value, Which is not what we want!!!

    # Optimizing for the best allocations
    # Setting the constraints (as per documentation:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    constraints ={"type": "eq", "fun": lambda allocs: 1 - np.sum(allocs)}  # type\:eq means equality
    # Means that the sum of the function is equal to 0. So 1 minus the sum of allocations == 0
    # Setting the bounds (as per the same documentation)
    bounds = [(0, 1) for i in range(n)]  # List comprehension to create bounds (0,1) for each symbol
    # Getting results from minimize function (as per documentation)
    res = spo.minimize(fun=optimization_function, x0=init_allocs, args=(prices,),
                       constraints=constraints, bounds=bounds, method="SLSQP",
                       options={"disp":False})
    minimzed_allocs = res.x

    # Getting Price Dataframes for Daily Portfolio Value Calculations
    normalized_prices, allocated_prices, pos_val, port_val = get_price_dfs(minimzed_allocs, prices)

    # add code here to find the allocations  		  	   		 	   			  		 			     			  	 
    # cr = Cumulative return
    # adr = Average daily return 
    # sddr = Standard deviation of daily return
    # sr = Sharpe Ratio
    cr, adr, sddr, sr = [
        get_cr(port_val),
        get_adr(port_val),
        get_sddr(port_val),
        get_sr(port_val)
    ]  # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   			  		 			     			  	 
    if gen_plot:
        # add code to plot here  		  	   		 	   			  		 			     			  	 
        df_temp = pd.concat(
            [port_val, normalized_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        create_plot(df_temp)

    return minimzed_allocs, cr, adr, sddr, sr


def test_code():
    """  		  	   		 	   			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	   			  		 			     			  	 
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio  		  	   		 	   			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics  		  	   		 	   			  		 			     			  	 
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		 	   			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	   			  		 			     			  	 
    test_code()
