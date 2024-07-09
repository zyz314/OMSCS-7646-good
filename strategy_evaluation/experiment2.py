from StrategyLearner import StrategyLearner
from indicators import (get_cr, 
                        get_adr, 
                        get_sddr, 
                        get_sr
                    )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def experiment(in_sd,in_ed,symbol,sv):
    commission = 0
    in_dates = pd.date_range(in_sd, in_ed)
    symbols = [symbol]
    results = dict()
    for impact in np.arange(start = 1, stop=13, step = 2)*0.005:
        Strat = StrategyLearner(impact=impact,commission=commission)
        Strat.add_evidence(symbol = symbol, sd = in_sd, ed = in_ed, sv = sv)
        in_trades = Strat.testPolicy_w_cash(symbol = symbol, sd = in_sd, ed = in_ed, sv = sv)
        in_prices = StrategyLearner.get_prices(in_dates, symbols)
        in_port_vals = StrategyLearner.create_port_vals(in_prices, in_trades, sv)
        results[impact]=in_port_vals
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")
    
    h = 7
    fig, ax = plt.subplots(figsize=(h*2,h))
    results_df.plot(ax=ax)
    plt.tight_layout()
    plt.savefig("p8_ex2.png")
    
    cr = results_df.apply(lambda port_val: get_cr(port_val), axis = 0)
    adr = results_df.apply(lambda port_val: get_adr(port_val), axis = 0)
    sddr = results_df.apply(lambda port_val: get_sddr(port_val), axis = 0)
    sr = results_df.apply(lambda port_val: get_sr(port_val), axis = 0)
    table = pd.concat([cr,adr,sddr,sr],axis=1)
    table.columns = ["CumRet","AvgDR","StdDR","SR"]
    table = table.T
    
    table.to_csv("p8_Experiment2_Table.csv")
    
    with open("p8_results.html","a") as f:
        # f.writelines(f"<html>")
        f.writelines(f"<h1>p8_results for Experiment 2 produced by mshihab6</h1>")
        f.writelines(table.to_html())
    
    # with open("Ex2_Results.txt","w") as f:
    #     f.writelines(table.to_latex())

def author():
    return "mshihab6"

if __name__ == "__main__":
    pass