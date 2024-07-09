import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from indicators import (
    BB_Pct_Digital as bbp,
    Stochastic_Oscillators_Digital as stosc, 
    MACD_Digital as macdd
    )
# Note: For all digital indicators, 0 is hold, 1 is buy, and -1 is sell
from marketsimcode import create_holdings
from util import get_data

class ManualStrategy(object):
    
    def __init__(self, impact=0.0, commission=0.0, verbose=False):
        """
        A manual learner that can learn (essenitally human coded rules) a trading policy using the same indicators used in StrategyLearner=.

        Parameters
            verbose (bool) - If “verbose” is True, your code can print out information for debugging.
                                If verbose = False your code should not generate ANY output.
            impact (float) - The market impact of each transaction, defaults to 0.0
            commission (float) - The commission amount charged, defaults to 0.0
            """
        self.impact = impact
        self.commission = commission
        self.verbose = verbose

    def add_evidence(
            self,
            symbol='IBM', 
            sd=datetime.datetime(2008, 1, 1, 0, 0), 
            ed=datetime.datetime(2009, 1, 1, 0, 0), 
            sv=10_000):
       
        # add evidence is used here for consistency with the Strategy Learner. In actuality, it is likely that 
        # this function does nothnig as the "rules" will have been coded by the developer of the class. If that is the case,
        # this this method might simply consist of a "pass" statement.

        """Trains your strategy learner over a given time frame.

        Parameters
            symbol (str) - The stock symbol to train on
            sd (datetime) - A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) - A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) - The starting value of the portfolio
        """
        pass

    def testPolicy(
            self,
            symbol='IBM', 
            sd=datetime.datetime(2009, 1, 1, 0, 0), 
            ed=datetime.datetime(2010, 1, 1, 0, 0), 
            sv=10000):
        """
        Tests your learner using data outside of the training data

        Parameters
            symbol (str) - The stock symbol that you trained on on
            sd (datetime) - A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) - A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) - The starting value of the portfolio
        Returns
            A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.

        Return type
            pandas.DataFrame
        """
        self.symbol = symbol
        self.sv = sv
        dates = pd.date_range(sd, ed)
        symbols = [symbol]
        prices = ManualStrategy.get_prices(dates, symbols)
        self.prices = prices
        orders = ManualStrategy.get_orders(prices)
        self.orders = orders
        trades = ManualStrategy.create_trades(prices, orders, sv, self.commission, self.impact)
        self.trades = trades
        return trades
        
    
    def author(self):
        """
        Returns
            The GT username of the student

        Return type
            str
        """
        return "mshihab6"
    
    # HELPER FUNCTIONS
    @staticmethod
    def get_prices(dates, symbols):
        prices = get_data(symbols, dates) # Need SPY to drop non-trading days
        prices = prices.dropna(subset=["SPY"])[symbols] # Just to be safe, drop nan values in SPY and only take required symboles
        prices = prices.fillna(method="ffill").fillna(method="bfill") # Just to be safe, fill forward and backwards
        prices["Cash"] = 1.0
        return prices
    
    @staticmethod
    def most_common_signal(row):
        # Count the occurrences of each value in the row, ignoring zeros
        counts = row[row != 0].value_counts()
        if counts.empty:
            return 0  # if all values are zero, return 0
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1] and counts.index[0] == -counts.index[1]:
            return 0  # if there's a tie between -1 and 1, return 0
        return counts.idxmax()  # return the most common non-zero value
    
    @staticmethod
    def get_orders(prices,
                   window = 20,
                   lower_band = 0.03,
                   upper_band = 1.00,
                   fast = 12,
                   slow = 26,
                   signal_window = 9,
                   macd_thresh = 0.7,
                   k=14,
                   d=4,
                   overbought_thresh=0.75,
                   oversold_thresh=0.2):
        symbol = prices.columns[0]

        bb = bbp(prices = prices.drop(columns="Cash"), window = window, lower_band = lower_band, upper_band = upper_band)
        macd = macdd(prices = prices.drop(columns="Cash"), fast = fast, slow = slow, signal_window = signal_window, macd_thresh = macd_thresh)
        stocs = stosc(prices = prices.drop(columns="Cash"), k = k, d = d, overbought_thresh = overbought_thresh, oversold_thresh = oversold_thresh)
        indicators = pd.concat([bb, macd, stocs],axis = 1)
        signal = indicators.apply(lambda signals: ManualStrategy.most_common_signal(signals), axis = 1).to_frame()
        signal = signal.rename(columns={0:symbol})
        orders = signal[signal[symbol]!=0].copy()
        return orders
    
    @staticmethod
    def buy_action(share_count:int):
        return 1000-share_count
    
    @staticmethod
    def sell_action(share_count:int):
        return -1000 - share_count
    
    @staticmethod
    def create_trades(prices, orders, sv, commission, impact):
        trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        trades.loc[trades.index.min(),"Cash"] = sv
        symbol = prices.columns[0]
        for date,action in orders.iterrows():
            total_shares = trades.loc[:date,symbol].sum()
            value = prices.loc[date][symbol]
            if action[symbol] == 1:
                shares = ManualStrategy.buy_action(total_shares)
            if action[symbol] == -1:
                shares = ManualStrategy.sell_action(total_shares)
            cash = value * shares
            deductions = (cash * impact) + commission
            trades.loc[date,symbol] = trades.loc[date,symbol] + shares
            trades.loc[date,"Cash"] = trades.loc[date,"Cash"] + (-cash - deductions)
        return trades
    
    def create_port_vals(self):    
        holdings = create_holdings(self.sv,self.trades)
        holdings = holdings.copy()
        values = self.prices * holdings
        port_vals = values.sum(axis=1)
        port_vals = port_vals/port_vals.iloc[0]
        return port_vals
    
    def create_benchmark_port_vals(self):
        orders = self.prices.iloc[[0]].drop(columns="Cash").copy()
        orders[self.symbol] = 1
        
        trades = ManualStrategy.create_trades(self.prices, orders, self.sv, self.commission, self.impact)
        
        holdings = create_holdings(self.sv,trades)
        holdings = holdings.copy()
        values = self.prices * holdings
        port_vals = values.sum(axis=1)
        port_vals = port_vals/port_vals.iloc[0]
        return port_vals
    
    @staticmethod
    def create_chart(symbol, trades, port_vals, bench_vals, in_sample, fig, ax):
        buy_dates = [str(d.date()) for d in trades[trades[symbol]>0].index]
        sell_dates = [str(d.date()) for d in trades[trades[symbol]<0].index]

        # fig, ax = plt.subplots()
        port_vals.plot(ax=ax, label = "Manual Strategy", color = "red")
        bench_vals.plot(ax=ax, label = "Benchmark", color = "purple")
        ax.legend()
        ax.set_title(f"{'In Sample' if in_sample else 'Out of Sample'} Manual Strategy vs Benchmark")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.grid(which="both", axis="x", linestyle=":", visible=True)

        y_min,y_max = ax.get_ylim()
        for line in buy_dates:
            ax.vlines(x=line,ymin=y_min,ymax=y_max, colors="blue",linestyles="dotted")
        for line in sell_dates:
            ax.vlines(x=line,ymin=y_min,ymax=y_max, colors="black",linestyles="dotted")

        # plt.tight_layout()
        # plt.savefig(f"p8_e1_man_{'in' if in_sample else 'out'}.png")
        # return fig, ax
    