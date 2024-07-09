""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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

import random as rand

import numpy as np

# 0: blank space. 
# 1: an obstacle. 
# 2: the starting location for the robot. 
# 3: the goal location. 
# 5: quicksand.

class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider. In a 10x10 problem, it would be 100 states
    :type num_states: int
    :param num_actions: The number of actions available. Since our robot can traverse up ,down, left, right, we have 4 actions.
    :type num_actions: int
    # Note: Qtable dimensions will be num_states (rows) x num_actions (columns)
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    # Note: How much do you trust new information?
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    # Note: What is the value of future rewards?
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    # Note Force the learner to explore
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    # Note: Reduce the rar since as time progresses, we become more certain of our choices
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    # Note: 0 means do not impliment dyna. 
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """
    def __init__(
        self,
        num_states: int = 100,
        num_actions: int = 4,
        alpha: float = 0.2,
        gamma: float = 0.9,
        rar: float = 0.5,
        radr: float = 0.99,
        dyna: int = 0,
        verbose: bool = False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        # Initialize Required Values
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Qtable = np.zeros(shape=(num_states, num_actions))
        # Dyna Specific Values
        self.dyna = dyna
        if dyna > 0:
            self.TCtable = np.full(shape=(num_states, num_actions, num_states), fill_value=1e-10) # To avoid dividing by zero error. 3D array since it is T[s,a,s_prime]
            self.Rtable = np.zeros(shape=(num_states, num_actions))
            self.previous = list()
            # self.previous = np.array([])
            # self.previous = dict()
            # self.previous = set()

    def querysetstate(self, s):
        # Note: I think this is for setting the very first state
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        # Setting the inital state
        self.s = s
        # Getting the initial action
        self.a = self.new_action(state = s) # Utilize best information later on rather than random decision
        # action = np.random.randint(low=0,high=self.num_actions) # Thought it was doing better than best action, but it turns out I am an idiot...
        if self.verbose:
            print(f"s = {s}, a = {self.a}")
        return self.a

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        # To make my life easier and make code look a little cleaner
        s = self.s
        a = self.a
        exp_tup = (s,a,s_prime,r)
        dyna = self.dyna
        alpha = self.alpha
        gamma = self.gamma
        
        # Algorithm to Follow:
        # 1) Update Q Table
        # 2) Check if Dyna is Required
        # 3) Run Dyna
        #   3.1) Update Rs
        #   3.2) Update TCs
        #   3.3) Get T to infer s_prime (at later step)
        #   3.4) For number of iterations specified in self.dyna
        #       3.4.1) Select random s
        #       3.4.2) Select random a
        #       3.4.3) Infer s_prime from calculated T
        #       3.4.4) Update Rs
        #       3.4.5) Update Q Table
        # Note: 3.4.1 to 3.4.5 were too slow and instead used Suttons Approach in 8.2 of his book
        # 4) Set self.s to s_prime (from query input not dyna)
        # 5) Calculate new best/random action from new state and self.rar
        # 6) Set self.a to action (from step 2)
        # 7) Return self.a
        
        # Learn from action taken by updating QTable
        self.Qtable[s][a] = self.update_q(experience_tuple=exp_tup, alpha=alpha,gamma=gamma)
              
        # ! This is where Dyna should go
        if dyna > 0: self.run_dyna(experience_tuple=exp_tup, dyna=dyna) # Suttons approach in 8.2 of his book
        else:
            if self.verbose: print("No Dyna")
        
        # Find new action (whether random or best next step)
        action = self.new_action(state = s_prime)
        self.rar *= self.radr # Drop the rar by radr
        
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        
        # Assign the new state and action to qlearner
        self.s = s_prime
        self.a = action
        
        return action

    def update_q(self, experience_tuple, alpha, gamma):
        s,a,s_prime,r = experience_tuple
        updated_q = ((1 - alpha) * self.Qtable[s][a]) + \
                            (alpha * (r + (gamma * self.Qtable[s_prime,np.argmax(self.Qtable[s_prime])])))
        return updated_q

    def calculate_T(self):
        # No longer used, but kept just in case
        # Thanks to the wonderful people who taught me about keep_dims in np.sum!
        # https://stackoverflow.com/questions/39441517/in-numpy-sum-there-is-parameter-called-keepdims-what-does-it-do
        T = self.TCtable/self.TCtable.sum(axis=2,keepdims=True)
        return T
    
    def update_R(self, experience_tuple, alpha):
        # No longer used, but kept just in case
        s,a,_,r = experience_tuple
        R = ((1-alpha)*self.Rtable[s][a]) + (alpha*r)
        return R

    def new_action(self, state):
        random_action_chance = np.random.random() <= self.rar # Do we take a random action
        if random_action_chance:
            if self.verbose: print("Random Action")
            return np.random.randint(low=0,high=self.num_actions)
        else:
            new_action = np.argmax(self.Qtable[state])
            if self.verbose: print(f"Action = {new_action}")
            return new_action

    def run_dyna(self, experience_tuple, dyna):
        self.previous.append(experience_tuple)
        previous = rand.choices(self.previous,k=dyna)
        self.run_halucinations(previous)
    
    def run_halucinations(self, previous_experiences):
        for experience_tuple in previous_experiences:
            self.Qtable[experience_tuple[0]][experience_tuple[1]] = self.update_q(experience_tuple=experience_tuple,alpha=self.alpha,gamma=self.gamma)
    
    def author(self):
        return "mshihab6"

def author():
    return "mshihab6"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
