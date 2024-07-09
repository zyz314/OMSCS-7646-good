import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import typing
import math
from util import get_data

def author():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """
    return "mshihab6"  # Change this to your user ID

def SMA(prices: pd.DataFrame, window: int = 50)->pd.DataFrame:
    """A function that returns the Simple Moving Average based on user input

    Args:
        prices (pd.DataFrame): Prices DataFrame
        window (int): How far back to look for the SMA. Recommended to fill in yourself
                      Usually 50 for short and 200 for long. These numbers are used for the golden cross method

    Returns:
        pd.DataFrame: A dataframe with the simple moving avearge calculated
    """
    offset_days = dt.timedelta(days=window*2)
    dates = pd.date_range(prices.index.min()-offset_days,prices.index.max())
    prices_window = get_data(prices.columns, dates)
    prices_window = prices_window.dropna(subset=["SPY"])
    prices_window = prices_window[prices.columns]
    prices_window = prices_window.fillna(method="ffill").fillna(method="bfill")
    
    return prices_window.rolling(window).mean()
def EMA(prices: pd.DataFrame, window: int = 10)-> pd.DataFrame:
    """A function that returns the Exponential Moving Average based on user input

    Args:
        prices (pd.DataFrame): Prices Dataframe
        window (int): How far back to look for the EMA. Defaults to 10, but has been known to be 50, 200

    Returns:
        pd.DataFrame: A dataframe with the exponential moving avearge calculated
    """
    offset_days = dt.timedelta(days=window*3)
    dates = pd.date_range(prices.index.min()-offset_days,prices.index.max())
    prices_window = get_data(prices.columns, dates)
    prices_window = prices_window.dropna(subset=["SPY"])
    prices_window = prices_window[prices.columns]
    prices_window = prices_window.fillna(method="ffill").fillna(method="bfill")
    
    return prices_window.ewm(span=window, adjust=False, min_periods=window).mean()

# Crosses
def Generic_Cross(prices: pd.DataFrame, short:int = 50, long:int = 200, ma_func:typing.Callable = SMA)->pd.DataFrame:
    """Produces a chart for the golden cross using SMA}

    Args:
        prices (pd.DataFrame): Prices DataFrame
        short (int, optional): The look back period for short periods. Defaults to 50.
        long (int, optional): The look back period for long periods. Defaults to 200.
        ma_func(Callable, optional): The moving average function of choice: Defaults to SMA
    
    Returns:
        pd.DataFrame: A dataframe with the MA_Short/MA_Long - 1. Subtracting 1 to make crossing point = 0 rather than 1
    """
    ma_short = ma_func(prices, short)
    ma_long  = ma_func(prices, long)
    cross = (ma_short/ma_long)-1
    return cross
def Golden_Cross(prices:pd.DataFrame, short:int=50, long:int=200)->pd.DataFrame:
    """Uses Crossing of MA data to calculate Golden Cross. Good for detecting Buy Opportunities

    Args:
        prices (pd.DataFrame): Prices DataFrame
        short (int, optional): The look back period for short periods. Defaults to 50.
        long (int, optional): The look back period for long periods. Defaults to 200.
    
    Returns:
        pd.DataFrame: A dataframe with the Golden Cross Points
    """
    cross = Generic_Cross(prices=prices, short=short, long=long).dropna(how="all")
    g_cross = (cross>=0) & (cross.shift(1)<0)
    g_cross = g_cross.astype(int) # 0 = False, 1 = True
    return g_cross    
def Death_Cross(prices:pd.DataFrame, short:int=50, long:int=200)->pd.DataFrame:
    """Uses Crossing of MA data to calculate Death Cross. Possible Sell Point?

    Args:
        prices (pd.DataFrame): Prices DataFrame
        short (int, optional): The look back period for short periods. Defaults to 50.
        long (int, optional): The look back period for long periods. Defaults to 200.
    
    Returns:
        pd.DataFrame: A dataframe with the Death Cross Points
    """
    cross = Generic_Cross(prices=prices, short=short, long=long).dropna(how="all")
    d_cross = (cross<=0) & (cross.shift(1)>0)
    d_cross = d_cross.astype(int) # 0 = False, 1 = True
    return d_cross    
def Cross_Digital(prices:pd.DataFrame, short:int=50, long:int=200)->pd.DataFrame:
    g_cross = Golden_Cross(prices=prices, short=short, long=long).loc[prices.index.min():]
    d_cross = Death_Cross(prices=prices, short=short, long=long).loc[prices.index.min():]*-1
    
    signals = g_cross+d_cross
    return signals

# BB% (You need BB to calculate BB%)
def Bollinger_Bands(prices: pd.DataFrame, window: int=20)-> tuple:
    """A function to return the upper and lower Bollinger Bands

    Args:
        prices (pd.DataFrame): Prices Dataframe
        window (int): How far back to look for the BB

    Returns:
        tuple: A tuple where element 0 is the lower bands and element 1 is the upper bands
    """
    offset_days = dt.timedelta(days=window*2)
    dates = pd.date_range(prices.index.min()-offset_days,prices.index.max())
    prices_window = get_data(prices.columns, dates)
    prices_window = prices_window.dropna(subset=["SPY"])
    prices_window = prices_window[prices.columns]
    prices_window = prices_window.fillna(method="ffill").fillna(method="bfill")
    
    prices_std = prices_window.rolling(window).std()
    prices_sma = SMA(prices_window, window)
    lower_band = prices_sma-(2*prices_std)
    upper_band = prices_sma+(2*prices_std)
    return lower_band, upper_band
def BB_Pct(prices: pd.DataFrame, window: int=20)->pd.DataFrame:
    """Calculating the Bollinger Bands Value as a percent

    Args:
        price (pd.DataFrame): Prices Dataframe
        window (int): How far back to look for the BB.
                      Used when calling Bollinger_Bands(prices: pd.DataFrame, window: int)

    Returns:
        pd.DataFrame: A dataframe with the prices relative position to the BB as a % where 0% is on the lower band and 100% is the upper band
    """
    offset_days = dt.timedelta(days=window*2)
    dates = pd.date_range(prices.index.min()-offset_days,prices.index.max())
    prices_window = get_data(prices.columns, dates)
    prices_window = prices_window.dropna(subset=["SPY"])
    prices_window = prices_window[prices.columns]
    prices_window = prices_window.fillna(method="ffill").fillna(method="bfill")
    
    bands = Bollinger_Bands(prices, window)
    return (prices_window - bands[0])/(bands[1] - bands[0])

def BB_Pct_Digital(prices: pd.DataFrame, window: int=20, lower_band = 0, upper_band = 1)->pd.DataFrame:
    bbands = BB_Pct(prices=prices, window = window).loc[prices.index.min():]
    # Breakout: When the price recrosses back into the envelope, it is a signal 
    # to buy (when crossing from underneath) 
    buy_signals = (bbands.shift(1)<lower_band) & (bbands>=lower_band).astype(int)
    # or sell (when crossing from up top).
    sell_signals = ((bbands.shift(1)>upper_band) & (bbands<=upper_band).astype(int))*-1
    signals = (buy_signals+sell_signals)
    return signals

# Moving Average Convergence Divergence (MACD)
def MACD(prices: pd.DataFrame, fast: int=12, slow: int=26, signal_window:int = 9)-> tuple:
    """A function that calculates Moving Average Convergence Divergence

    Args:
        prices (pd.DataFrame): Prices Dataframe
        fast (int, optional): How far back to look for MACD fast. Defaults to 12.
        slow (int, optional): How far back to look for MACD slow. Defaults to 26.
        signal (int, optional): How far back to look for MACD signal. Defaults to 9.

    Returns:
        tuple: A tuple with 3 dataframes: The MACD, Signal Line, and Histogram
    """
    ema_fast = EMA(prices=prices, window=fast)
    ema_slow = EMA(prices=prices, window=slow)
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust = False, min_periods = signal_window).mean()
    histogram = macd - signal
    return macd, signal, histogram
    #return macd.loc[prices.index.min():], signal.loc[prices.index.min():], histogram.loc[prices.index.min():]
def MACD_Digital(prices: pd.DataFrame, fast: int=12, slow: int=26, signal_window:int = 9, macd_thresh = 1)->pd.DataFrame:
    sd = prices.index.min()
    macd, signal, histogram = MACD(prices = prices, fast = fast, slow = slow, signal_window = signal_window)
    
    macd = macd.loc[sd:]
    signal = signal.loc[sd:]
    histogram = histogram.loc[sd:]
    
    buy_signals = ((macd < -macd_thresh) & (macd>signal) & (macd.shift(1)<=signal.shift(1))).astype(int)
    sell_signals = ((macd > macd_thresh) & ((macd<signal) & (macd.shift(1)>=signal.shift(1))).astype(int))*-1
    signals = buy_signals + sell_signals
    return signals

# Stochastic Oscillators
def Stochastic_Oscillators(prices: pd.DataFrame, k: int=14, d: int=3)-> tuple:
    """A function that calculates Stochastic Oscilators

    Args:
        prices (pd.DataFrame): Prices Dataframe
        k (int, optional): How far back to look for Stochastic Oscillators k. Defaults to 14.
        d (int, optional): How far back to look for Stochastic Oscillators d. Defaults to 3.

    Returns:
        tuple: A tuple with 2 dataframes: Stochastic Oscillators fast and Stochastic Oscillators slow
    """
    offset_days = dt.timedelta(days=k*3)
    dates = pd.date_range(prices.index.min()-offset_days,prices.index.max())
    lows = get_data(prices.columns, dates, colname="Low")
    lows = lows.dropna(subset=["SPY"])
    lows = lows[prices.columns]
    lows = lows.fillna(method="ffill").fillna(method="bfill")
    
    highs = get_data(prices.columns, dates, colname="High")
    highs = highs.dropna(subset=["SPY"])
    highs = highs[prices.columns]
    highs = highs.fillna(method="ffill").fillna(method="bfill")
    
    close = get_data(prices.columns, dates, colname="Close")
    close = close.dropna(subset=["SPY"])
    close = close[prices.columns]
    close = close.fillna(method="ffill").fillna(method="bfill")
    
    low_k_mins = lows.rolling(k).min()
    # low_k_mins = low_k_mins.loc[prices.index]
    high_k_maxs = highs.rolling(k).max()
    # high_k_maxs = high_k_maxs.loc[prices.index]
    
    stochastic_k = (close - low_k_mins)/(high_k_maxs - low_k_mins)
    stochastic_d = stochastic_k.rolling(d).mean()
    
    # Maybe returning the ratio between fast and moving average?
    return stochastic_k, stochastic_d
def Stochastic_Oscillators_Digital(prices: pd.DataFrame, k: int=14, d: int=3, overbought_thresh=0.8, oversold_thresh=0.2)->pd.DataFrame:
    # Note: K is Fast and D is slow
    sd = prices.index.min()
    stochastic_k, stochastic_d = Stochastic_Oscillators(prices=prices, k=k, d=d)
    stochastic_k = stochastic_k.loc[sd:]
    stochastic_d = stochastic_d.loc[sd:]
    # Algorithm
    # if fast and slow above overbought line and fast < slow: sell
    sell_signals = (((stochastic_k>overbought_thresh) & (stochastic_d>overbought_thresh) & (stochastic_k.shift(1)>stochastic_d.shift(1)) & (stochastic_k<stochastic_d)).astype(int))*-1
    # else if fast and slow below oversold line and fast > slow: buy
    buy_signals = ((stochastic_k<oversold_thresh) & (stochastic_d<oversold_thresh) & (stochastic_k.shift(1)<stochastic_d.shift(1)) & (stochastic_k>stochastic_d).astype(int))
    # else: do nothing
    signals = (buy_signals+sell_signals)
    return signals


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

def create_table(in_port_vals, oos_port_vals, rfr=0):
    # In Sample Results
    cr = get_cr(in_port_vals)
    adr = get_adr(in_port_vals)
    sddr = get_sddr(in_port_vals)
    sr = get_sr(in_port_vals, risk_free_rate = rfr)
    # Out of Sample Results
    oos_cr = get_cr(oos_port_vals)
    oos_adr = get_adr(oos_port_vals)
    oos_sddr = get_sddr(oos_port_vals)
    oos_sr = get_sr(oos_port_vals, risk_free_rate = rfr)
    
    # Table Data Dictionary
    data = {
        "In Sample":[cr,adr,sddr,sr],
        "Out Of Sample":[oos_cr,oos_adr,oos_sddr,oos_sr],
    }
    
    # DataFrame
    results = pd.DataFrame(data)
    results.index = ["CumRet","AvgDR","StdDR","SR"]
    
    return results