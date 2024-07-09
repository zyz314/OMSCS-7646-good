""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import math  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
# this function should return a dataset (X and Y) that will work  		  	   		 	   			  		 			     			  	 
# better for linear regression than decision trees  		  	   		 	   			  		 			     			  	 
def best_4_lin_reg(seed=1489683273):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 	   			  		 			     			  	 
    :type seed: int  		  	   		 	   			  		 			     			  	 
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    np.random.seed(seed)  		  	   		 	   			  		 			     			  	 
    # x = np.random.randint(low=10,high=100,size=(100, 2))  		  	   		 	   			  		 			     			  	 
    # y = np.random.random(size=(100,)) * 200 - 100  		  	   		 	   			  		 			     			  	 
    # Here's is an example of creating a Y from randomly generated  		  	   		 	   			  		 			     			  	 
    # X with multiple columns  		  	   		 	   			  		 			     			  	 
    # y = x[:,0] + np.sin(x[:,1]) + x[:,2]**2 + x[:,3]**3  		  	   		 	   			  		 
    row_count = np.random.randint(low=10,high=1001)
    column_count = np.random.randint(low=2,high=11)
    x = np.random.random(size=(row_count, column_count)) * 300-50
    # y = mx+b
    m = np.random.randint(low=1,high=6)
    b = np.random.randint(low=-20,high=21)
    y = (m*x).sum(axis=1)+b			     
    # Since the model is linear, it will succeed on linearly created data			  	 
    return x, y  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def best_4_dt(seed=1489683273):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 	   			  		 			     			  	 
    :type seed: int  		  	   		 	   			  		 			     			  	 
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    np.random.seed(seed)  		  	   		 	   			  		 			     			  	 
    # x = np.zeros((100, 2))  		  	   		 	   			  		 			     			  	 
    # y = np.random.random(size=(100,)) * 200 - 100  		  	   		 	   			  		 			     			  	 
    row_count = np.random.randint(low=10,high=1001)
    column_count = np.random.randint(low=2,high=11)
    x = np.random.random(size=(row_count, column_count)) * 300-50
    
    # y = np.power(x,2).sum(axis=1) # 
    # y = np.power(x,3).sum(axis=1) # 
    # y = np.power(x,4).sum(axis=1) # 
    # y = np.log2(x).sum(axis=1) #
    # y = np.sin(x).sum(axis=1) #
    # y = np.cos(x).sum(axis=1) #
    # y = np.tan(x).sum(axis=1) #
    degrees = np.random.randint(low=2,high=5)
    # y = polynomial_2_to_5(x,degrees) #
    y = multi_function_gen(x)
    return x, y  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def polynomial_2_to_5(x,degrees):
    y = np.zeros(x.shape[0])
    for i in range(2,degrees+1):
        y += np.power(x,i).sum(axis=1)
    y += x.sum(axis=1)
    return y

def multi_function_gen(x):
    # functions = ["sin","cos","tan","pow2","pow3","pow4","log2","sum","sub"] #
    functions = ["sin","cos","tan","pow2","pow3","pow4"] #
    # functions = ["sin","cos","tan"] # 
    y = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        func = np.random.choice(functions)
        if func == "sin":
            y+=np.sin(x[:,i])
        if func == "cos":
            y+=np.cos(x[:,i])
        if func == "tan":
            y+=np.tan(x[:,i])
        if func == "pow2":
            y+=np.power(x[:,i],2)
        if func == "pow3":
            y+=np.power(x[:,i],3)
        if func == "pow4":
            y+=np.power(x[:,i],4)
        if func == "log2":
            y+=np.log2(x[:,i])
        if func == "sum":
            y+=np.sum(x[:,i])    
        if func == "sub":
            y+=np.sum(-x[:,i])
    return y
  		  	   		 	   			  		 			     			  	 
def author():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return "mshihab6"  # Change this to your user ID
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("they call me Tim.")  		  	   		 	   			  		 			     			  	 
