from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from indicators import create_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def experiment(params):
    symbol, man_in_port_vals, man_out_port_vals, strat_in_port_vals, strat_out_port_vals, in_bench_vals, out_bench_vals = params
    h = 7
    fig, ax = plt.subplots(nrows=2, figsize=(h*2,h))
    create_chart(symbol, man_in_port_vals, man_out_port_vals, strat_in_port_vals, strat_out_port_vals, in_bench_vals, out_bench_vals, ax)
    pass

def create_chart(symbol, man_in_port_vals, man_out_port_vals, strat_in_port_vals, strat_out_port_vals, in_bench_vals, out_bench_vals, ax):
    man_in_port_vals.plot(ax=ax[0], label = "Manual Strategy", color = "red")
    strat_in_port_vals.plot(ax=ax[0], label = "Strategy Learner", color = "blue")
    in_bench_vals.plot(ax=ax[0], label = "Benchmark", color = "purple")
    ax[0].legend()
    ax[0].set_title(f"In Sample Manual Strategy vs Strategy Learner vs Benchmark")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Normalized Value")
    ax[0].grid(which="both", axis="x", linestyle=":", visible=True)
    
    man_out_port_vals.plot(ax=ax[1], label = "Manual Strategy", color = "red")
    strat_out_port_vals.plot(ax=ax[1], label = "Strategy Learner", color = "blue")
    out_bench_vals.plot(ax=ax[1], label = "Benchmark", color = "purple")
    ax[1].legend()
    ax[1].set_title(f"Out of Sample Manual Strategy vs Strategy Learner vs Benchmark")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Normalized Value")
    ax[1].grid(which="both", axis="x", linestyle=":", visible=True)
    
    plt.tight_layout()
    plt.savefig("p8_ex1.png")
    pass

def author():
    return "mshihab6"

if __name__ == "__main__":
    pass