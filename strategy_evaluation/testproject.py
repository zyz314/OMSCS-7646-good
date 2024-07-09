from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner 
import experiment1 as ex1
import experiment2 as ex2
from indicators import create_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903687444 

def author():
    return "mshihab6"

np.random.seed(gtid())

def manual_part(in_sd,in_ed,oos_sd,oos_ed,symbol,impact,commission,sv):
    # The in-sample/testing period is from January 1, 2008, to December 31, 2009. 
    in_dates = pd.date_range(in_sd, in_ed)
    # The out-of-sample/testing period is from January 1, 2010, to December 31, 2011. 
    oos_dates = pd.date_range(oos_sd, oos_ed)
    symbols = [symbol]
    
    h=7
    fig, ax = plt.subplots(nrows=2, figsize=(h*2,h))

    ManStrat_in  = ManualStrategy(impact=impact, commission=commission)
    in_trades = ManStrat_in.testPolicy(symbol=symbol, 
            sd=in_sd, 
            ed=in_ed, 
            sv=sv)
    in_port_vals = ManStrat_in.create_port_vals()
    in_bench_vals = ManStrat_in.create_benchmark_port_vals()
    ManualStrategy.create_chart(symbol, in_trades, in_port_vals, in_bench_vals, in_sample=True, fig = fig, ax = ax[0])
    
    ManStrat_out = ManualStrategy(impact=impact, commission=commission)
    out_trades = ManStrat_out.testPolicy(symbol=symbol, 
            sd=oos_sd, 
            ed=oos_ed, 
            sv=sv)
    out_port_vals = ManStrat_out.create_port_vals()
    # print(out_port_vals, type(out_port_vals))
    out_bench_vals = ManStrat_out.create_benchmark_port_vals()
    ManualStrategy.create_chart(symbol, out_trades, out_port_vals, out_bench_vals, in_sample=False, fig = fig, ax = ax[1])
    
    plt.tight_layout()
    plt.savefig("p8_man.png") 
    
    ManStrat_results = create_table(in_port_vals, out_port_vals, rfr=0)
    Bench_results = create_table(in_bench_vals, out_bench_vals, rfr=0)
    
    ManStrat_results.to_csv("p8_ManualStrategy_Table.csv")
    Bench_results.to_csv("p8_Benchmark_Table.csv")
    with open("p8_results.html","w+") as f:
        f.writelines(f"<html>")
        f.writelines(f"<h1>p8_results for ManualStrategy produced by mshihab6</h1>")
        f.writelines(ManStrat_results.to_html())
        
    with open("p8_results.html","a") as f:
        f.writelines(f"<html>")
        f.writelines(f"<h1>p8_results for Benchmark produced by mshihab6</h1>")
        f.writelines(Bench_results.to_html())
    
    with open("ManualStrategy_Results.txt","w") as f:
        f.writelines(ManStrat_results.to_latex())
    
    with open("Bench_Results.txt","w") as f:
        f.writelines(Bench_results.to_latex())

    return in_trades, in_port_vals, in_bench_vals, out_trades, out_port_vals, out_bench_vals

def ml_part(in_sd,in_ed,oos_sd,oos_ed,symbol,impact,commission,sv):
    # The in-sample/testing period is from January 1, 2008, to December 31, 2009. 
    in_dates = pd.date_range(in_sd, in_ed)
    # The out-of-sample/testing period is from January 1, 2010, to December 31, 2011. 
    oos_dates = pd.date_range(oos_sd, oos_ed)
    symbols = [symbol]
    Strat = StrategyLearner(impact=impact,commission=commission)
    Strat.add_evidence(symbol = symbol, sd = in_sd, ed = in_ed, sv = sv)
    in_trades = Strat.testPolicy_w_cash(symbol = symbol, sd = in_sd, ed = in_ed, sv = sv)
    # in_trades.to_csv("test_trades_in.csv")
    
    h = 7
    fig, ax = plt.subplots(nrows=2, figsize=(h*2,h))
    
    in_prices = StrategyLearner.get_prices(in_dates, symbols)
    in_port_vals = StrategyLearner.create_port_vals(in_prices, in_trades, sv)
    in_bench_vals = StrategyLearner.create_benchmark_port_vals(in_prices, symbol, sv, commission, impact)
    # print(in_port_vals, type(in_port_vals))
    StrategyLearner.create_chart(symbol, in_trades, in_port_vals, in_bench_vals, True, fig, ax[0])
    
    oos_prices = StrategyLearner.get_prices(oos_dates, symbols)
    oos_trades = Strat.testPolicy_w_cash(symbol = symbol, sd = oos_sd, ed = oos_ed, sv = sv)
    # oos_trades.to_csv("test_trades_oos.csv")
    oos_port_vals = StrategyLearner.create_port_vals(oos_prices, oos_trades, sv)
    # print(oos_port_vals, type(oos_port_vals))
    oos_bench_vals = StrategyLearner.create_benchmark_port_vals(oos_prices, symbol, sv, commission, impact)
    StrategyLearner.create_chart(symbol, oos_trades, oos_port_vals, oos_bench_vals, False, fig, ax[1])
    
    plt.tight_layout()
    plt.savefig("p8_strat.png")
    
    Strat_results = create_table(in_port_vals, oos_port_vals, rfr=0)
    
    Strat_results.to_csv("p8_StrategyLearner_Table.csv")
    
    with open("p8_results.html","a") as f:
        # f.writelines(f"<html>")
        f.writelines(f"<h1>p8_results for StrategyLearner produced by mshihab6</h1>")
        f.writelines(Strat_results.to_html())
    
    # with open("StrategyLearner_Results.txt","w") as f:
    #     f.writelines(Strat_results.to_latex())
        
    return in_trades, in_port_vals, in_bench_vals, oos_trades, oos_port_vals, oos_bench_vals

def main():
    symbol = "JPM"
    # The in-sample period is from January 1, 2008, to December 31, 2009. 
    in_sd = datetime.datetime(2008,1,1)
    in_ed = datetime.datetime(2009,12,31)
    # The out-of-sample/testing period is from January 1, 2010, to December 31, 2011. 
    oos_sd = datetime.datetime(2010,1,1)
    oos_ed = datetime.datetime(2011,12,31)
    impact=0.005
    commission=9.95
    sv=10_000
    man_in_trades, man_in_port_vals, man_in_bench_vals, man_out_trades, man_out_port_vals, man_out_bench_vals = manual_part(in_sd,in_ed,oos_sd,oos_ed,symbol,impact,commission,sv)
    strat_in_trades, strat_in_port_vals, strat_in_bench_vals, strat_out_trades, strat_out_port_vals, strat_out_bench_vals = ml_part(in_sd,in_ed,oos_sd,oos_ed,symbol,impact,commission,sv)
    ex1_params = (symbol, man_in_port_vals, man_out_port_vals, strat_in_port_vals, strat_out_port_vals, strat_in_bench_vals, strat_out_bench_vals)
    ex1.experiment(ex1_params)
    ex2.experiment(in_sd,in_ed,symbol,sv)
if __name__ == "__main__":
    main()