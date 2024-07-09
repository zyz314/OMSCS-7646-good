import numpy as np
from scipy.stats import mode # For Mode

class BagLearner(object):

    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        """
        This is a Bootstrap Aggregataion Learner (BagLearner). You will need to properly implement this class as necessary.

        Parameters
            learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
            kwargs            - Keyword arguments that are passed on to the learner's constructor and they can vary according to the learner
            bags (int)        - The number of learners you should train using Bootstrap Aggregation. 
                                If boost is true, then you should implement boosting (optional implementation).
            verbose (bool)    - If “verbose” is True, your code can print out information for debugging.
                                If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        """
        if kwargs is None:
            kwargs = {}
        # self.learner = learner(**kwargs)  # Define the Learner using the keyword arguments passed
        self.learner = learner  # Define the Learner using the keyword arguments passed
        self.kwargs = kwargs # Keep track of keyword arguments
        self.bags = bags  # Store the bag count
        self.boost = boost  # Store the boolean boost value
        self.verbose = verbose  # Store the boolean verbose value
        self.ensemble = np.array(
            [self.learner(**kwargs) for _ in range(self.bags)])  # create numpy array of learners to make an ensemble

    def create_bags(self, data_x, data_y):
        n = len(data_x)
        bags = list()
        bag_ys = list()
        indexes = np.random.choice(n, size=n, replace=True)  # creates a numpy array of shape (n) of row numbers per bag
        bags = np.array(data_x[indexes,:])
        # using each row numbers to create bags of shape (n,c) where c is the column number in data_x
        bag_ys = np.array(data_y[indexes])  # using each row numbers to create the ys for each bag of shape (n)
        return bags, bag_ys  # return the created bags and their respective ys

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        Parameters
            data_x (numpy.ndarray) - A set of feature values used to train the learner
            data_y (numpy.ndarray) - The value we are attempting to predict given the X data
        """
        for learner in self.ensemble:
            bags, bag_ys = self.create_bags(data_x, data_y)  # get all the bags and their respective ys
            learner.add_evidence(bags, bag_ys)  # build learner from the bags created above
            # No need to make a build learner since each learner should have it already

    def author(self):
        """
        Returns
            The GT username of the student

        Return type
            str
        """
        return "mshihab6"

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) - A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
        """
        predictions = []
        for learner in self.ensemble:
            predictions.append(learner.query(points))
        # print(predictions)
        return mode(np.array(predictions), axis=0)[0]

