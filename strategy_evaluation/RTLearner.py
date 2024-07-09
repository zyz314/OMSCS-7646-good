import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        This is a Random Tree Learner (RTLearner). You will need to properly implement this class as necessary.

        Parameters
            leaf_size (int)  - Is the maximum number of samples to be aggregated at a leaf
            verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                               If verbose = False your code should not generate ANY output. 
                               When we test your code, verbose will be False.
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.learner = None  # Will be used when building and querying

    # Added for debugging purposes
    def __repr__(self):
        return (f"RTLearner(leaf_size={self.leaf_size}, verbose={self.verbose})")

    # Added for debugging purposes
    def __str__(self):
        return (f"Tree:\n{self.learner}")
        return (f"RTLearner(leaf_size={self.leaf_size}, verbose={self.verbose})")

    def add_evidence(self, data_x, data_y):
        """
            Add training data to learner

            Parameters
                data_x (numpy.ndarray) - A set of feature values used to train the learner
                data_y (numpy.ndarray) - The value we are attempting to predict given the X data
        """
        # data_x & data_y since the API requirements ask for it like that
        self.learner = self.build_tree(data_x, data_y).reshape([-1, 4])

    def find_best_feature(self, data_x):
        i = np.random.randint(0, data_x.shape[1])
        return i

    def find_SplitVal(self, data_x, i):
        # random_row_1 = np.random.randint(0, data_x.shape[0])
        # random_row_2 = np.random.randint(0, data_x.shape[0])
        # while random_row_2 == random_row_1:
        #     random_row_2 = np.random.randint(0, data_x.shape[0])
        # SplitVal = (data_x[random_row_1,i]+data_x[random_row_2,i])/2
        SplitVal = np.median(data_x[:, i])
        return SplitVal

    @staticmethod
    def my_mode(data_y):
        unique_vals, counts = np.unique(data_y, return_counts=True)
        max_count = np.max(counts)

        # Check if there's a tie between two or more values
        if np.sum(counts == max_count) > 1:
            return 0
        else:
            return unique_vals[np.argmax(counts)]

    def build_tree(self, data_x, data_y):
        """
            Building learner with given data using recurrsion

            Parameters
                data_x (numpy.ndarray) - A set of feature values used to train the learner
                data_y (numpy.ndarray) - The value we are attempting to predict given the X data
                
            Returns
                The numpy array that represents our tree

            Return type
                np.ndarray
        """
        # data_x & data_y since the API requirements ask for it like that

        # if there leaf_size number of rows, return
        if data_x.shape[0] <= self.leaf_size:
            return np.array([-1, RTLearner.my_mode(data_y), np.nan, np.nan])

        # if there is only one value in y (example: y=[1,1,1,1,1,1]), return
        if len(np.unique(data_y)) == 1:
            return np.array([-1, data_y[0], np.nan, np.nan])

        # else: determine best feature i to split on
        i = self.find_best_feature(data_x)

        # Find the Split Value [median]
        SplitVal = self.find_SplitVal(data_x, i)
        LeftSplitCond = data_x[:, i] <= SplitVal
        # On the off chance that every condition in LeftSplitCond is all true or all false, return
        if (np.all((LeftSplitCond == False)) | np.all((LeftSplitCond == True))):
            return np.array([-1, RTLearner.my_mode(data_y), np.nan, np.nan])
        RightSplitCond = data_x[:, i] > SplitVal

        # Build Left Side of Tree
        leftdata_x, leftdata_y = data_x[LeftSplitCond], data_y[LeftSplitCond]
        lefttree = self.build_tree(leftdata_x, leftdata_y)

        # Build Right Side of Tree
        rightdata_x, rightdata_y = data_x[RightSplitCond], data_y[RightSplitCond]
        righttree = self.build_tree(rightdata_x, rightdata_y)

        # Return Full Tree
        root = np.array([i, SplitVal, 1, lefttree.reshape([-1, 4]).shape[0] + 1])
        # Note: np.append was not working as intended. Used np.concatenate
        tree = np.concatenate([root, lefttree, righttree])
        return tree

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
        ypred = np.array([])
        for point in points:
            i = 0
            node = 0
            while i >= 0:
                i = int(self.learner[node][0])
                if i < 0: break
                SplitVal = self.learner[node][1]
                SplitCon = point[i] <= SplitVal
                if SplitCon:
                    node += self.learner[node][2]
                else:
                    node += self.learner[node][3]
                node = int(node)
            ypred = np.append(ypred, self.learner[node][1])
        return ypred
