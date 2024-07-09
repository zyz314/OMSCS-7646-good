""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
import random

import pandas as pd
import numpy as np
import util as ut

import BagLearner as bl
import RTLearner as rt

from indicators import (
    BB_Pct as bbpf,
    Stochastic_Oscillators as stoscf, 
    MACD as macdf
    )
from marketsimcode import create_holdings

class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)
        self.commission = commission
        
        # pred_y = self.learner.query(train_x)

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        
        N = 5
        ybuy = 1.02
        ysell = 0.98
        
        # add your code to do learning here
        dates = pd.date_range(sd,ed)
        symbols = [symbol]
        
        prices = StrategyLearner.get_prices(dates, symbols)
        train_X = np.array(StrategyLearner.get_indicators(prices = prices, symbol = symbol))
        train_y = np.array(StrategyLearner.get_y(prices = prices, symbol = symbol, N = N, ybuy = ybuy, ysell = ysell, impact = self.impact))
        
        self.learner.add_evidence(train_X, train_y)
        
        """# example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # example use with new colname
        volume_all = ut.get_data(
            syms, dates, colname="Volume"
        )  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)"""

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100_000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        prices = StrategyLearner.get_prices(dates,symbols)
        test_X = np.array(StrategyLearner.get_indicators(prices = prices, symbol = symbol))
        pred_y = self.learner.query(test_X)[0]
        test_orders = pd.DataFrame(pred_y, index = prices.index)
        test_orders = test_orders.rename(columns = {0: symbol})
        test_orders = test_orders[test_orders[symbol]!=0]
        test_trades = StrategyLearner.create_trades(prices, test_orders, sv, self.commission, self.impact)
        return test_trades.drop(columns="Cash")
        """# here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
        trades.values[:, :] = 0  # set them all to nothing
        trades.values[0, :] = 1000  # add a BUY at the start
        trades.values[40, :] = -1000  # add a SELL
        trades.values[41, :] = 1000  # add a BUY
        trades.values[60, :] = -2000  # go short from long
        trades.values[61, :] = 2000  # go long from short
        trades.values[-1, :] = -1000  # exit on the last day
        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)
        if self.verbose:
            print(prices_all)
        return trades"""

    def testPolicy_w_cash(
        self,
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100_000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        prices = StrategyLearner.get_prices(dates,symbols)
        test_X = np.array(StrategyLearner.get_indicators(prices = prices, symbol = symbol))
        pred_y = self.learner.query(test_X)[0]
        test_orders = pd.DataFrame(pred_y, index = prices.index)
        test_orders = test_orders.rename(columns = {0: symbol})
        test_orders = test_orders[test_orders[symbol]!=0]
        test_trades = StrategyLearner.create_trades(prices, test_orders, sv, self.commission, self.impact)
        return test_trades
    
    def author(self):
        return "mshihab6"
    
    @staticmethod
    def get_indicators(prices, symbol,
                   window = 20,
                   fast = 12,
                   slow = 26,
                   signal_window = 9,
                   k=14,
                   d=4):    
        bbp = bbpf(prices = prices, window = window).loc[prices.index.min():]
        bbp = bbp.rename(columns={symbol:"bbp"})
        ind_0 = bbp

        macd, signal, _ = macdf(prices = prices, fast = fast, slow = slow, signal_window = signal_window)
        macd = macd.loc[prices.index.min():]
        signal = signal.loc[prices.index.min():]
        ind_1 = (signal/macd).rename(columns = {symbol:"macd"})

        k_df,d_df = stoscf(prices = prices, k = k, d = d)
        ind_2 = (d_df/k_df).rename(columns = {symbol:"stocs"})
        ind_2 = ind_2.loc[prices.index.min():]

        indicators = pd.concat([ind_0,ind_1,ind_2],axis = 1)
        
        return indicators
    
    @staticmethod
    def get_prices(dates, symbols):
        prices = ut.get_data(symbols, dates) # Need SPY to drop non-trading days
        prices = prices.dropna(subset=["SPY"])[symbols] # Just to be safe, drop nan values in SPY and only take required symboles
        prices = prices.fillna(method="ffill").fillna(method="bfill") # Just to be safe, fill forward and backwards
        # prices["Cash"] = 1.0
        return prices

    @staticmethod
    def convert_y(rets, symbol, ybuy, ysell, impact):
        conditions = [
            rets > (ybuy+impact),
            rets < (ysell-impact)
        ]
        choices = [1, -1]
        return np.select(conditions, choices, default=0)

    @staticmethod
    def get_y(prices, symbol, N, ybuy, ysell, impact):
        rets = prices.shift(-N)/prices
        rets['discrete'] = StrategyLearner.convert_y(np.array(rets), symbol, ybuy, ysell, impact)
        return rets
    
    @staticmethod
    def buy_action(share_count:int):
        return 1000-share_count
    
    @staticmethod
    def sell_action(share_count:int):
        return -1000 - share_count
    
    @staticmethod
    def create_trades(prices, orders, sv, commission, impact):
        prices["Cash"] = 1
        trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        trades.loc[trades.index.min(),"Cash"] = sv
        symbol = prices.columns[0]
        for date,action in orders.iterrows():
            total_shares = trades.loc[:date,symbol].sum()
            value = prices.loc[date][symbol]
            if action[symbol] == 1:
                shares = StrategyLearner.buy_action(total_shares)
            if action[symbol] == -1:
                shares = StrategyLearner.sell_action(total_shares)
            if shares == 0: continue
            cash = value * shares
            deductions = (cash * impact) + commission
            trades.loc[date,symbol] = trades.loc[date,symbol] + shares
            trades.loc[date,"Cash"] = trades.loc[date,"Cash"] + (-cash - deductions)
        return trades

    @staticmethod
    def create_port_vals(prices, trades, sv):    
        prices["Cash"] = 1
        holdings = create_holdings(sv,trades)
        holdings = holdings.copy()
        values = prices * holdings
        port_vals = values.sum(axis=1)
        port_vals = port_vals/port_vals.iloc[0]
        return port_vals
    
    @staticmethod
    def create_benchmark_port_vals(prices, symbol, sv, commission, impact):
        orders = prices.iloc[[0]].copy()#.drop(columns="Cash").copy()
        orders[symbol] = 1
        
        trades = StrategyLearner.create_trades(prices, orders, sv, commission, impact)
        
        holdings = create_holdings(sv,trades)
        holdings = holdings.copy()
        values = prices * holdings
        port_vals = values.sum(axis=1)
        port_vals = port_vals/port_vals.iloc[0]
        return port_vals
    
    
    @staticmethod
    def create_chart(symbol, trades, port_vals, bench_vals, in_sample, fig, ax):
        buy_dates = [str(d.date()) for d in trades[trades[symbol]>0].index]
        sell_dates = [str(d.date()) for d in trades[trades[symbol]<0].index]

        # fig, ax = plt.subplots()
        port_vals.plot(ax=ax, label = "Strategy Learner", color = "red")
        bench_vals.plot(ax=ax, label = "Benchmark", color = "purple")
        ax.legend()
        ax.set_title(f"{'In Sample' if in_sample else 'Out of Sample'} Strategy Learner vs Benchmark")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.grid(which="both", axis="x", linestyle=":", visible=True)

        y_min,y_max = ax.get_ylim()
        for line in buy_dates:
            ax.vlines(x=line,ymin=y_min,ymax=y_max, colors="blue",linestyles="dotted")
        for line in sell_dates:
            ax.vlines(x=line,ymin=y_min,ymax=y_max, colors="black",linestyles="dotted")

if __name__ == "__main__":
    print("One does not simply think up a strategy")
    
