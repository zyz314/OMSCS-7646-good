""""""
"""Assess a betting strategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def author():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """
    return "mshihab6"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	   			  		 			     			  	 
    :rtype: int  		  	   		 	   			  		 			     			  	 
    """
    return 903687444  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		 	   			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   			  		 			     			  	 

    :param win_prob: The probability of winning  		  	   		 	   			  		 			     			  	 
    :type win_prob: float  		  	   		 	   			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	   			  		 			     			  	 
    :rtype: bool  		  	   		 	   			  		 			     			  	 
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

#  API Requirements above. Do Note Edit


def betting_strategy(win_probability):
    episode_winnings = 0
    winnings = np.full(shape=(1001,), fill_value=80)  # Will increment by 1 when win even after a series of losses
    # Therefore, it should always end in 80
    spin = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            if spin >= 1001:
                return winnings
            winnings[spin] = episode_winnings
            won = get_spin_result(win_probability)
            spin += 1
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2
    return winnings


def realistic_betting_strategy(win_probability: float, bankroll_amount: float = None):
    episode_winnings = 0
    winnings = np.full(shape=(1001,), fill_value=80)
    spin = 0
    while (episode_winnings < 80) and (episode_winnings > -bankroll_amount):
        won = False
        bet_amount = 1
        while not won:
            if spin >= 1001:
                # print("returning end")
                return winnings
            if episode_winnings <= -bankroll_amount:
                winnings[spin:] = episode_winnings
                # print("returning loss")
                return winnings
            winnings[spin] = episode_winnings
            won = get_spin_result(win_probability)
            spin += 1
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = min(bet_amount * 2, bankroll_amount + episode_winnings)  # it is bankroll+episode_winnings
                # since the case where this will be relevant is when the episode_winnings will be < 0 and close to -256
    # print("returning win")
    return winnings


def experiment(num_episodes: int, win_probability: float, bankroll_problem: bool = False,
               bankroll_amount: float = None):
    episodes = np.zeros((num_episodes, 1001))
    for i in range(num_episodes):
        if bankroll_problem:
            episodes[i] = realistic_betting_strategy(win_probability, bankroll_amount=bankroll_amount)
        else:
            episodes[i] = betting_strategy(win_probability)
    return episodes


def aggregate_experiments(experiment_array: np.ndarray, aggregation: str = "mean"):
    if aggregation == "mean":
        return np.mean(experiment_array, axis=0)
    if aggregation == "median":
        return np.median(experiment_array, axis=0)
    if aggregation == "std":
        return np.std(experiment_array, axis=0)
    return experiment_array


def calculate_win_prob(experiment_array: np.ndarray):
    # Since the win condition is reaching 80, anything that isn't 80, is considered not a win
    loss_prob = (experiment_array[:, -1] < 80).sum() / len(experiment_array)
    win_prob = 1 - loss_prob
    return win_prob


def calculate_win_count(experiment_array: np.ndarray):
    win_count = (experiment_array[:, -1] == 80).sum()
    return win_count


def average_num_spins_to_win(experiment_array: np.ndarray):
    first_wins = np.array([])
    for arr in experiment_array:
        try:
            first_wins = np.append(first_wins, np.argwhere(arr == 80)[0][0])
        except IndexError:
            first_wins = np.append(first_wins, np.nan)
    return (np.round(np.nanmean(first_wins), 2),
            np.round(np.nanmedian(first_wins), 2),
            np.round(np.nanstd(first_wins), 2))


def calculate_below_threshold(experiment_array: np.ndarray, threshold: int = -256):
    experiment_array_min = np.min(experiment_array, axis=1)
    return len(np.where(experiment_array_min < threshold)[0])


def calculate_winnings_expectation(experiment_array: np.ndarray):
    expectation_array = np.unique(experiment_array[:, -1], return_counts=True)
    expectation = (expectation_array[0] * expectation_array[1] / expectation_array[1].sum()).sum()
    return expectation_array, expectation


def create_experiment_plot(title: str, experiment_array: np.ndarray, agg: str = None, std: bool = False, xlim=(0, 300),
                           ylim=(-256, 100)):
    fig, ax = plt.subplots()
    # ax.set_xlim(0, 300)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("Spin Number")
    # ax.set_ylim(-256, 100)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel("Total Winnings")
    if agg is None:
        ax.set_title(f"Figure {title.split('_')[0]}: {len(experiment_array):,} Episodes")
        std = False
        for i, episode in enumerate(experiment_array):
            ax.plot(range(1001), episode, label=f"Episode {i}")
    if agg in ["mean", "median"]:
        ax.set_title(
            f"Figure {title.split('_')[0]}: {len(experiment_array):,} Episodes with {agg} aggregation. Win_Prob: {calculate_win_prob(experiment_array):.2f}")
        aggregated_array = aggregate_experiments(experiment_array, aggregation=agg)
        aggregated_array_mean = np.mean(aggregated_array)
        ax.plot(range(1001), aggregated_array, color="black", label="Experiment")
        ax.plot(range(1001), [aggregated_array_mean] * 1001, color="grey", linestyle=(0, (2, 5)),
                label=f"Average {agg} of Experiment: {aggregated_array_mean:.2f}")
        if std:
            std = aggregate_experiments(experiment_array, aggregation="std")
            ax.plot(range(1001), aggregated_array + std, color="black", linestyle="--")
            ax.plot(range(1001), aggregated_array - std, color="black", linestyle="--")
            ax.fill_between(range(1001), aggregated_array, aggregated_array + std, alpha=0.2, color="green")
            ax.fill_between(range(1001), aggregated_array, aggregated_array - std, alpha=0.2, color="red")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"images/fig_{title}.png")


def test_code():
    """
    Method to test your code
    """
    win_prob = 0.4737
    # There are numbers from 1 to 36 (alternating in color based on a specific set of conditions)
    # and two green sections labeled 0 and 00. The odds of winning are 1 1/9 to 1 https://en.wikipedia.org/wiki/Roulette
    # Which when converted using https://www.covers.com/tools/odds-converter is around 47.37%
    # ~18/38 since there are 38 slots, and 18 are red, 18 are black, and 2 green (0 and 00)
    # What are the odds of hitting red or black in roulette? The odds of hitting red
    # or black in American roulette are 47.37%. https://www.casino.org/roulette/odds/
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    ex1: np.ndarray = experiment(10, win_prob)
    create_experiment_plot("1", ex1, agg=None, std=False)
    # create_experiment_plot("1_mean", ex1, agg="mean", std=True)
    # create_experiment_plot("1_median", ex1, agg="median", std=True)
    print("fig_1 complete")
    ex2: np.ndarray = experiment(1_000, win_prob)
    create_experiment_plot("2", ex2, agg="mean", std=True)
    # create_experiment_plot("2_no_agg", ex2, agg=None, std=False) # Ignore, meant for sanity check
    # create_experiment_plot("2_mean_zoomout", ex2, agg="mean", std=True, ylim=(-4500,250)) # Ignore, meant for sanity check
    print("fig_2 complete")
    create_experiment_plot("3", ex2, agg="median", std=True)
    print("fig_3 complete")
    # np.savetxt("ex2.csv", ex2, delimiter=",")
    ex4: np.ndarray = experiment(1_000, win_prob, bankroll_problem=True, bankroll_amount=256)
    create_experiment_plot("4", ex4, agg="mean", std=True)
    # create_experiment_plot("4_no_agg", ex4, agg=None, std=False) # Ignore, meant for sanity check
    # create_experiment_plot("4_mean_zoom", ex4, agg="mean", std=True, ylim=(-60,10)) # Ignore, meant for sanity check
    print("fig_4 complete")
    create_experiment_plot("5", ex4, agg="median", std=True)
    print("fig_5 complete")
    # np.savetxt("ex4.csv", ex4, delimiter=",")

    # Stat Calculations
    ex1_bank: np.ndarray = experiment(10, win_prob, bankroll_problem=True, bankroll_amount=256)
    create_experiment_plot("1_bank", ex1_bank, agg=None, std=False)
    # np.savetxt("ex1_bank.csv", ex1_bank, delimiter=",")
    thresholds = [0, -50, -100, -256, -500, -1_000, -16_000, -1_000_000, -4_000_000, -5_000_000]
    below_dict = {
        "ex1_10": list(),
        "ex2_10": list(),
        "ex1_1000": list(),
        "ex2_1000": list(),
    }
    win_count_dict = {}
    spin_to_win_dict = {}
    expectation_winnings_dict = {}

    for t in thresholds:
        below_dict["ex1_10"].append(calculate_below_threshold(ex1, t))
        below_dict["ex2_10"].append(calculate_below_threshold(ex1_bank, t))
        below_dict["ex1_1000"].append(calculate_below_threshold(ex2, t))
        below_dict["ex2_1000"].append(calculate_below_threshold(ex4, t))
    below_df = pd.DataFrame(below_dict, index=thresholds)

    win_count_dict["ex1_10"] = [calculate_win_count(ex1)]
    win_count_dict["ex2_10"] = [calculate_win_count(ex1_bank)]
    win_count_dict["ex1_1000"] = [calculate_win_count(ex2)]
    win_count_dict["ex2_1000"] = [calculate_win_count(ex4)]
    win_count_df = pd.DataFrame(win_count_dict)

    spin_to_win_dict["ex1_10"] = average_num_spins_to_win(ex1)
    spin_to_win_dict["ex2_10"] = average_num_spins_to_win(ex1_bank)
    spin_to_win_dict["ex1_1000"] = average_num_spins_to_win(ex2)
    spin_to_win_dict["ex2_1000"] = average_num_spins_to_win(ex4)
    spin_to_win_df = pd.DataFrame(spin_to_win_dict, index=["Avg Spin Count", "Median Spin Count", "STD"])

    ex1_10_exp_array = calculate_winnings_expectation(ex1)[0]
    expectation_winnings_dict["ex1_10"] = [calculate_winnings_expectation(ex1)[1]]
    ex2_10_exp_array = calculate_winnings_expectation(ex1_bank)[0]
    expectation_winnings_dict["ex2_10"] = [calculate_winnings_expectation(ex1_bank)[1]]
    ex1_1000_exp_array = calculate_winnings_expectation(ex2)[0]
    expectation_winnings_dict["ex1_1000"] = [calculate_winnings_expectation(ex2)[1]]
    ex2_1000_exp_array = calculate_winnings_expectation(ex4)[0]
    expectation_winnings_dict["ex2_1000"] = [calculate_winnings_expectation(ex4)[1]]
    expectation_winnings_df = pd.DataFrame(expectation_winnings_dict, index=["Winning Expectations"])

    renaming_dict = {"ex1_10": "Experiment 1: 10 Episodes",
                     "ex2_10": "Experiment 2: 10 Episodes",
                     "ex1_1000": "Experiment 1: 1000 Episodes",
                     "ex2_1000": "Experiment 2: 1000 Episodes",
                     }
    below_df = below_df.rename(columns=renaming_dict)
    win_count_df = win_count_df.rename(columns=renaming_dict).T.rename(columns={0: "Win Count"})
    spin_to_win_df = spin_to_win_df.rename(columns=renaming_dict).T
    expectation_winnings_df = expectation_winnings_df.rename(columns=renaming_dict).T

    # p1_results.html creation
    with open("p1_results.html", "w") as f:
        f.writelines(f"<html>")
        f.writelines(f"<h1>p1_results for martingale.py produced by {author()}: {gtid()}</h1>")

        f.writelines("<h2>Table 1: Winning Expectation per Experiment </h2>")
        f.writelines(expectation_winnings_df.to_html())
        f.writelines("<br><b>Experiment 1: 10 Episodes</b>")
        value_str = "<p>There were:<br>"
        for value_pair in list(zip(ex1_10_exp_array[0], ex1_10_exp_array[1])):
            value_str += f"   - {value_pair[1]:>3} instances of the {int(value_pair[0]):>4} = {value_pair[0] * value_pair[1] / 10:>8.4f}<br>"
        value_str += f"Expected Value = {(ex1_10_exp_array[0] * ex1_10_exp_array[1] / ex1_10_exp_array[1].sum()).sum()}</p>"
        f.writelines(value_str)

        f.writelines("<b>Experiment 2: 10 Episodes</b>")
        value_str = "<p>There were:<br>"
        for value_pair in list(zip(ex2_10_exp_array[0], ex2_10_exp_array[1])):
            value_str += f"   - {value_pair[1]:>3} instances of the {int(value_pair[0]):>4} = {value_pair[0] * value_pair[1] / 10:>8.4f}<br>"
        value_str += f"Expected Value = {(ex2_10_exp_array[0] * ex2_10_exp_array[1] / ex2_10_exp_array[1].sum()).sum()}</p>"
        f.writelines(value_str)

        f.writelines("<b>Experiment 1: 1000 Episodes</b>")
        value_str = "<p>There were:<br>"
        for value_pair in list(zip(ex1_1000_exp_array[0], ex1_1000_exp_array[1])):
            value_str += f"   - {value_pair[1]:>3} instances of the {int(value_pair[0]):>4} = {value_pair[0] * value_pair[1] / 1000:>8.4f}<br>"
        value_str += f"Expected Value = {(ex1_1000_exp_array[0] * ex1_1000_exp_array[1] / ex1_1000_exp_array[1].sum()).sum()}</p>"
        f.writelines(value_str)

        f.writelines("<b>Experiment 2: 1000 Episodes</b>")
        value_str = "<p>There were:<br>"
        for value_pair in list(zip(ex2_1000_exp_array[0], ex2_1000_exp_array[1])):
            value_str += f"   - {value_pair[1]:>3} instances of the {int(value_pair[0]):>4} = {value_pair[0] * value_pair[1] / 1000:>8.4f}<br>"
        value_str += f"Expected Value = {(ex2_1000_exp_array[0] * ex2_1000_exp_array[1] / ex2_1000_exp_array[1].sum()).sum()}</p>"
        f.writelines(value_str)

        f.writelines("<h2>Table 2: Number of experiment episodes that reached the $80 win condition</h2>")
        f.writelines(win_count_df.to_html())

        f.writelines("<h2>Table 3: Number of experiment episodes below a certain winnings threshold </h2>")
        f.writelines("<i>Note: The count shown in the table uses the less than operator</i>")
        f.writelines(below_df.to_html())

        f.writelines("<h2>Table 4: Number of spins to win per experiment</h2>")
        f.writelines(spin_to_win_df.to_html())

        f.writelines(f"</html>")

    # Extra Charts
    print("Beginning extras")
    extra_ex1: np.ndarray = experiment(1_000, win_prob, bankroll_problem=True, bankroll_amount=200_000)
    extra_ex1: np.ndarray = experiment(1_000, win_prob, bankroll_problem=True, bankroll_amount=200_000)
    create_experiment_plot("6", extra_ex1, agg="mean", std=True)
    extra_ex2: np.ndarray = experiment(1_000, win_prob, bankroll_problem=True, bankroll_amount=4_000_000)
    create_experiment_plot("7", extra_ex2, agg="mean", std=True)


if __name__ == "__main__":
    test_code()
