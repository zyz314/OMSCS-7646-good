""""""
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
"""

import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

gtid = 903687444
np.random.seed(gtid)


# https://docs.python.org/dev/library/time.html#time.process_time
# https://www.geeksforgeeks.org/how-to-find-size-of-an-object-in-python/
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
def RMSE(y_true, y_pred):
    # rmse = math.sqrt(((y_true - y_pred) ** 2).sum() / y_true.shape[0])
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def MAE(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def ME(y_true, y_pred):
    # ME is Max_Error
    me = np.max(np.abs(y_true - y_pred))
    return me


def R2(y_true, y_pred):
    # Code for eps found here: https://stackoverflow.com/questions/29709614/python-epsilon-is-not-the-smallest-number
    eps = sys.float_info.min
    r2 = 1 - (((y_true - y_pred) ** 2).sum()) / max(eps, ((y_true - y_true.mean()) ** 2).sum())
    return max(0, r2)


def generate_error_plot(df, x_axis, x_axis_label, figure_name, title):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(df[x_axis], df.rmse_in, label="In Sample Error")
    ax[0].plot(df[x_axis], df.rmse_out, label="Out Sample Error")
    ax[0].set_ylabel("RMSE")
    ax[0].minorticks_on()
    ax[0].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[0].legend()

    ax[1].plot(df[x_axis], df.corr_in, label="In Sample Error")
    ax[1].plot(df[x_axis], df.corr_out, label="Out Sample Error")
    ax[1].set_xlabel(x_axis_label)
    ax[1].set_ylabel("Correlation")
    ax[1].minorticks_on()
    ax[1].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[1].legend()

    plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(f"{figure_name}.png", format="png")


def generate_times_plot(df, x_axis, x_axis_label, figure_name, title):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(df[x_axis], df.train_ptime, label="Train Time")
    ax[0].set_ylabel("Training Process Time")
    ax[0].minorticks_on()
    ax[0].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[0].legend()

    ax[1].plot(df[x_axis], df.query_in_ptime, label="In Sample Query Time")
    ax[1].plot(df[x_axis], df.query_out_ptime, label="Out Sample Query Time")
    ax[1].set_xlabel(x_axis_label)
    ax[1].set_ylabel("Query Process Time")
    ax[1].minorticks_on()
    ax[1].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[1].legend()

    plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(f"{figure_name}.png", format="png")


def experiment_1(train_x, train_y, test_x, test_y, export= False):
    """
    Research and discuss overfitting as observed in the experiment. (Use the dataset Istanbul.csv with DTLearner).
    Support your assertion with graphs/charts. (Do not use bagging in Experiment 1). At a minimum, the following
    question(s) that must be answered in the discussion:

    - Does overfitting occur with respect to leaf_size?
    - For which values of leaf_size does overfitting occur? Indicate the starting point and the direction of overfitting
    Support your answer in the discussion or analysis. Use RMSE as your metric for assessing overfitting.
    -Include a discussion of what overfitting is, why it occurs, why it is important, and how it is mitigated.
    """
    max_leaf_size = 100
    train_times = list()
    train_ptimes = list()

    rmse_in_array = list()
    corr_in_array = list()
    query_in_times = list()
    query_in_ptimes = list()

    rmse_out_array = list()
    corr_out_array = list()
    query_out_times = list()
    query_out_ptimes = list()

    leaf_sizes = np.arange(start=1, stop=max_leaf_size + 1, step=1)
    for leaf_size in leaf_sizes:
        # Perform Training
        tree = dt.DTLearner(leaf_size=leaf_size)
        start_time, start_ptime = time.time(), time.process_time()
        tree.add_evidence(train_x, train_y)  # Training
        end_time, end_ptime = time.time(), time.process_time()
        train_time, train_ptime = end_time - start_time, end_ptime - start_ptime
        train_times.append(train_time)
        train_ptimes.append(train_ptime)
        # Following the template provided by the original code
        # Query In Sample
        start_time, start_ptime = time.time(), time.process_time()
        pred_y = tree.query(train_x)  # In Sample Querying
        end_time, end_ptime = time.time(), time.process_time()
        # Get In Sample Results
        query_time, query_ptime = end_time - start_time, end_ptime - start_ptime
        rmse_in = RMSE(train_y, pred_y)
        corr_in = np.corrcoef(pred_y, y=train_y)[0, 1]
        # Store In Sample Results
        rmse_in_array.append(rmse_in)
        corr_in_array.append(corr_in)
        query_in_times.append(query_time)
        query_in_ptimes.append(query_ptime)

        # Query Out Sample
        start_time, start_ptime = time.time(), time.process_time()
        pred_y = tree.query(test_x)
        end_time, end_ptime = time.time(), time.process_time()
        query_time, query_ptime = end_time - start_time, end_ptime - start_ptime
        # Get Out Sample Results
        rmse_out = RMSE(test_y, pred_y)
        corr_out = np.corrcoef(pred_y, y=test_y)[0, 1]
        # Store Out Sample Results
        rmse_out_array.append(rmse_out)
        corr_out_array.append(corr_out)
        query_out_times.append(query_time)
        query_out_ptimes.append(query_ptime)

    # Collect Results in DataFrame for Plotting and CSV Extraction (debugging)
    results = (pd.DataFrame([leaf_sizes, rmse_in_array, corr_in_array, rmse_out_array, corr_out_array, train_times,
                             train_ptimes, query_in_times, query_in_ptimes, query_out_times, query_out_ptimes],
                            index=["leaf_size", "rmse_in", "corr_in", "rmse_out", "corr_out", "train_time",
                                   "train_ptime",
                                   "query_in_time", "query_in_ptime", "query_out_time", "query_out_ptime"]).T)

    if export: results.to_csv("e1.csv")

    # Create Plots
    title = "RMSE Error vs Leaf Size for DTLearner"
    generate_error_plot(results, "leaf_size", "Leaf Size", "experiment1_error", title)
    title = "Process Time vs Leaf Size for DTLearner"
    generate_times_plot(results, "leaf_size", "Leaf Size", "experiment1_time", title)


def experiment_2(train_x, train_y, test_x, test_y, export=False):
    """
    Research and discuss the use of bagging and its effect on overfitting. (Again, use the dataset Istanbul.csv with
    DTLearner.) Provide charts to validate your conclusions. Use RMSE as your metric. At a minimum, the following
    questions(s) must be answered in the discussion.

    -Can bagging reduce overfitting with respect to leaf_size?
    -Can bagging eliminate overfitting with respect to leaf_size?

    To investigate these questions, choose a fixed number of bags to use and vary leaf_size to evaluate. If there is
    overfitting, indicate the starting point and the direction of overfitting. Support your answer in the discussion
    or analysis.
    """
    max_leaf_size = 100
    train_times = list()
    train_ptimes = list()

    rmse_in_array = list()
    corr_in_array = list()
    query_in_times = list()
    query_in_ptimes = list()

    rmse_out_array = list()
    corr_out_array = list()
    query_out_times = list()
    query_out_ptimes = list()

    leaf_sizes = np.arange(start=1, stop=max_leaf_size + 1, step=1)
    for leaf_size in leaf_sizes:
        # Perform Training
        bagger = bl.BagLearner(dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20)
        start_time, start_ptime = time.time(), time.process_time()
        bagger.add_evidence(train_x, train_y)
        end_time, end_ptime = time.time(), time.process_time()
        train_time, train_ptime = end_time - start_time, end_ptime - start_ptime
        train_times.append(train_time)
        train_ptimes.append(train_ptime)
        # Following the template provided by the original code
        # Query In Sample
        start_time, start_ptime = time.time(), time.process_time()
        pred_y = bagger.query(train_x)
        end_time, end_ptime = time.time(), time.process_time()
        # Get In Sample Results
        query_time, query_ptime = end_time - start_time, end_ptime - start_ptime
        rmse_in = RMSE(train_y, pred_y)
        corr_in = np.corrcoef(pred_y, y=train_y)[0, 1]
        # Store In Sample Results
        rmse_in_array.append(rmse_in)
        corr_in_array.append(corr_in)
        query_in_times.append(query_time)
        query_in_ptimes.append(query_ptime)

        # Query Out Sample
        start_time, start_ptime = time.time(), time.process_time()
        pred_y = bagger.query(test_x)
        end_time, end_ptime = time.time(), time.process_time()
        query_time, query_ptime = end_time - start_time, end_ptime - start_ptime
        # Get Out Sample Results
        rmse_out = RMSE(test_y, pred_y)
        corr_out = np.corrcoef(pred_y, y=test_y)[0, 1]
        # Store Out Sample Results
        rmse_out_array.append(rmse_out)
        corr_out_array.append(corr_out)
        query_out_times.append(query_time)
        query_out_ptimes.append(query_ptime)

    # Collect Results in DataFrame for Plotting and CSV Extraction (debugging)
    results = (pd.DataFrame([leaf_sizes, rmse_in_array, corr_in_array, rmse_out_array, corr_out_array, train_times,
                             train_ptimes, query_in_times, query_in_ptimes, query_out_times, query_out_ptimes],
                            index=["leaf_size", "rmse_in", "corr_in", "rmse_out", "corr_out", "train_time",
                                   "train_ptime",
                                   "query_in_time", "query_in_ptime", "query_out_time", "query_out_ptime"]).T)

    if export: results.to_csv("e2.csv")

    # Create Plots
    title = "RMSE Error vs Leaf Size for BagLearner(DTLearner)"
    generate_error_plot(results, "leaf_size", "Leaf Size", "experiment2_error", title)
    title = "Process Time vs Leaf Size for BagLearner(DTLearner)"
    generate_times_plot(results, "leaf_size", "Leaf Size", "experiment2_time", title)


# https://stackoverflow.com/questions/43925337/matplotlib-returning-a-plot-object
def create_average_function(df,column,y_lim,window_value, fig, ax):
    df["d"+column+"_in"].rolling(window_value).mean().plot(ax=ax, label = "DT In")
    df["d"+column+"_out"].rolling(window_value).mean().plot(ax=ax, label = "DT Out")
    df["r"+column+"_in"].rolling(window_value).mean().plot(ax=ax, label = "RT In")
    df["r"+column+"_out"].rolling(window_value).mean().plot(ax=ax, label = "RT Out")
    ax.hlines([df["d"+column+"_in"].mean(), df["d"+column+"_out"].mean(), df["r"+column+"_in"].mean(), df["r"+column+"_out"].mean()],0,100,["blue","orange","green","red"],"dotted")
    ax.set_xlim(0,100)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Leaf Size")
    ax.set_ylabel(column)
    ax.set_title(column)
    ax.grid(visible=True, which="both",axis="x",linestyle="dotted")
    ax.legend()


def experiment_3(train_x, train_y, test_x, test_y, export= False):
    """
    Quantitatively compare “classic” decision trees (DTLearner) versus random trees (RTLearner). For this part of the
    report, you must conduct new experiments; do not use the results of Experiment 1. Importantly, RMSE, MSE,
    correlation, and time to query are not allowed as metrics for this experiment.

    Provide at least two new quantitative measures in the comparison.

    - Using two similar measures that illustrate the same broader metric does not count as two separate measures. (
    Note: Do not use two measures for the accuracy or use the same measurement for two different attributes – e.g.,
    time to train and time to query are both considered a use of the “time” metric.)
    - Provide charts to support your conclusions.

    At a minimum, the following question(s) must be answered in the discussion.

    - In which ways is one method better than the other?
    - Which learner had better performance (based on your selected measures) and why do you think that was the case?
    - Is one learner likely to always be superior to another (why or why not)?

    Note: Metrics that have been used in prior terms include Mean Absolute Error (MAE), Coefficient of Determination
    (R-Squared), Mean Absolute Percentage Error (MAPE), Maximum Error (ME), Time, and Space. In addition, please feel
    free to explore the use of other metrics you discover.
    """
    max_leaf_size = 100
    # Provide Metric Arrays
    dMAE_in, dMAE_out = list(), list()
    rMAE_in, rMAE_out = list(), list()
    dR2_in, dR2_out = list(), list()
    rR2_in, rR2_out = list(), list()
    dME_in, dME_out = list(), list()
    rME_in, rME_out = list(), list()

    leaf_sizes = np.arange(start=1, stop=max_leaf_size + 1, step=1)
    for leaf_size in leaf_sizes:
        # Create Trees
        dTree = dt.DTLearner(leaf_size=leaf_size)
        rTree = rt.RTLearner(leaf_size=leaf_size)
        # Train Trees
        dTree.add_evidence(train_x, train_y)
        rTree.add_evidence(train_x, train_y)
        # Query In Sample
        dT_pred_y = dTree.query(train_x)
        rT_pred_y = rTree.query(train_x)
        # Get In Sample Results
        dmae, rmae = MAE(train_y, dT_pred_y), MAE(train_y, rT_pred_y)
        dr2, rr2 = R2(train_y, dT_pred_y), R2(train_y, rT_pred_y)
        dme, rme = ME(train_y, dT_pred_y), ME(train_y, rT_pred_y)
        # Store In Sample Results
        dMAE_in.append(dmae)
        rMAE_in.append(rmae)
        dR2_in.append(dr2)
        rR2_in.append(rr2)
        dME_in.append(dme)
        rME_in.append(rme)

        # Query Out Sample
        dT_pred_y = dTree.query(test_x)
        rT_pred_y = rTree.query(test_x)
        # Get Out Sample Results
        dmae, rmae = MAE(test_y, dT_pred_y), MAE(test_y, rT_pred_y)
        dr2, rr2 = R2(test_y, dT_pred_y), R2(test_y, rT_pred_y)
        dme, rme = ME(test_y, dT_pred_y), ME(test_y, rT_pred_y)
        # Store Out Sample Results
        dMAE_out.append(dmae)
        rMAE_out.append(rmae)
        dR2_out.append(dr2)
        rR2_out.append(rr2)
        dME_out.append(dme)
        rME_out.append(rme)

    # Collect Results in DataFrame for Plotting and CSV Extraction (debugging)
    results = (pd.DataFrame([leaf_sizes, dMAE_in, rMAE_in, dR2_in, rR2_in, dME_in, rME_in, dMAE_out, rMAE_out,
                             dR2_out, rR2_out, dME_out, rME_out],
                            index=["leaf_size", "dMAE_in", "rMAE_in", "dR2_in", "rR2_in", "dME_in", "rME_in",
                                   "dMAE_out", "rMAE_out", "dR2_out", "rR2_out", "dME_out", "rME_out"]).T)

    if export: results.to_csv("e3.csv")

    # Create Charts
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 4))
    # DT Results on first row
    ax[0].plot(results.leaf_size, results.dMAE_in, label="DT In")
    ax[0].plot(results.leaf_size, results.dMAE_out, label="DT Out")
    ax[1].plot(results.leaf_size, results.dR2_in, label="DT In")
    ax[1].plot(results.leaf_size, results.dR2_out, label="DT Out")
    ax[2].plot(results.leaf_size, results.dME_in, label="DT In")
    ax[2].plot(results.leaf_size, results.dME_out, label="DT Out")

    # RT Results
    ax[0].plot(results.leaf_size, results.rMAE_in, label="RT In")
    ax[0].plot(results.leaf_size, results.rMAE_out, label="RT Out")
    ax[1].plot(results.leaf_size, results.rR2_in, label="RT In")
    ax[1].plot(results.leaf_size, results.rR2_out, label="RT Out")
    ax[2].plot(results.leaf_size, results.rME_in, label="RT In")
    ax[2].plot(results.leaf_size, results.rME_out, label="RT Out")

    # Chart Details
    ax[0].set_title("MAE")
    ax[0].set_xlabel("Leaf Size")
    ax[0].set_ylabel("MAE")
    ax[0].legend()
    ax[0].minorticks_on()
    ax[0].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[1].set_title("R2")
    ax[1].set_xlabel("Leaf Size")
    ax[1].set_ylabel("R2")
    ax[1].legend()
    ax[1].minorticks_on()
    ax[1].grid(which="both", axis="x", linestyle=":", visible=True)
    ax[2].set_title("ME")
    ax[2].set_xlabel("Leaf Size")
    ax[2].set_ylabel("ME")
    ax[2].legend()
    ax[2].minorticks_on()
    ax[2].grid(which="both", axis="x", linestyle=":", visible=True)
    fig.suptitle("Decision Tree vs Random Tree")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("experiment3.png", format="png")
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 4))
    window_value = 10
    create_average_function(results, "MAE", (0,0.007), window_value, fig, ax[0])
    create_average_function(results, "R2", (0,1), window_value, fig, ax[1])
    create_average_function(results, "ME", (0,0.05), window_value, fig, ax[2])
    fig.suptitle(f"Rolling Average of metrics over {window_value} leaf sizes")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("experiment3_rolling_avg.png", format="png")

# Was going to use as original function for experiments, but found no use for it.
# Kept around just in case
def evaluate_in_sample(learner, train_x, train_y):
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}, {RMSE(train_y, pred_y)}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")


# Was going to use as original function for experiments, but found no use for it.
# Kept around just in case
def evaluate_out_of_sample(learner, test_x, test_y):
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = RMSE(test_y, pred_y)
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}, {RMSE(test_y, pred_y)}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def run_experiments(train_x, train_y, test_x, test_y, verbose=False):
    if verbose:
        start_time, full_time = time.time(), time.time()  # Using time.time to make sure file doesn't take 10 minutes
    experiment_1(train_x, train_y, test_x, test_y, export = verbose)
    if verbose:
        print(f"Exp. 1 time: {time.time() - start_time}")  # Each Experiment Run Time
        start_time = time.time()
    experiment_2(train_x, train_y, test_x, test_y, export = verbose)
    if verbose:
        print(f"Exp. 2 time: {time.time() - start_time}")  # Each Experiment Run Time
        start_time = time.time()
    experiment_3(train_x, train_y, test_x, test_y, export = verbose)
    if verbose:
        print(f"Exp. 3 time: {time.time() - start_time}")  # Each Experiment Run Time
        print(f"Exp. Suite time: {time.time() - full_time}")  # Full Experiment Suite Run Time


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    # Changed to use the with keyword as an open without a close MAY cause issues
    # (yet to be seen, but better be safe than sorry)
    with open(sys.argv[1]) as inf:
        data = np.array(
            [list(map(str, s.strip().split(","))) for s in inf.readlines()]  # float was causing issues, changed to str
        )
    # Check if file is Istanbul and act accordingly
    file_name = sys.argv[1].split("/")[1]
    if file_name == "Istanbul.csv":
        data = data[1:, 1:]

    # compute how much of the data is training and testing  		  	   		 	   			  		 			     			  	 
    N = data.shape[0]
    train_split = np.random.choice(a=N, size=int(0.6 * N), replace=False)
    train_rows = np.full(N, False)
    train_rows[train_split] = True

    # separate out training and testing data  		  	   		 	   			  		 			     			  	 
    train_x = data[train_rows, 0:-1].astype(float)  # recast to floats since original split is str
    train_y = data[train_rows, -1].astype(float)  # recast to floats since original split is str
    test_x = data[~train_rows, 0:-1].astype(float)  # recast to floats since original split is str
    test_y = data[~train_rows, -1].astype(float)  # recast to floats since original split is str

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # create a learner and train it  		  	   		 	   			  		 			     			  	 
    # learner = lrl.LinRegLearner(verbose=False)  # create a LinRegLearner
    # learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=20)  # create a BagLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())  # Print Author to make sure function still works
    # evaluate_in_sample(learner, train_x, train_y)  # Kept around JUST IN CASE
    # evaluate_out_of_sample(learner, test_x, test_y)  # Kept around JUST IN CASE

    # print()
    run_experiments(train_x, train_y, test_x, test_y, verbose=False)
