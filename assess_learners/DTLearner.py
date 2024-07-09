import numpy as np

class DTLearner(object):

    def __init__(self,leaf_size=1, verbose=False):
        """
        This is a Decision Tree Learner (DTLearner). You will need to properly implement this class as necessary.

        Parameters
            leaf_size (int)  - Is the maximum number of samples to be aggregated at a leaf
            verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                               If verbose = False your code should not generate ANY output. 
                               When we test your code, verbose will be False.
        """
        self.leaf_size=leaf_size
        self.verbose=verbose
        self.learner = None # Will be used when building and querying

    # Added for debugging purposes
    def __repr__(self):
        return(f"DTLearner(leaf_size={self.leaf_size}, verbose={self.verbose})")
    
    # Added for debugging purposes
    def __str__(self):
        return(f"DTLearner(leaf_size={self.leaf_size}, verbose={self.verbose})")
    
    def add_evidence(self, data_x, data_y):
        """
            Add training data to learner

            Parameters
                data_x (numpy.ndarray) - A set of feature values used to train the learner
                data_y (numpy.ndarray) - The value we are attempting to predict given the X data
        """
        # data_x & data_y since the API requirements ask for it like that
        self.learner = self.build_tree(data_x,data_y).reshape([-1,4])
    

    def find_best_feature(self, data_x, data_y):
        # Step 1: Correlation Matrix
        correlation_matrix = np.corrcoef(data_x,data_y,rowvar=False)
        # np.corrcoef to find the correlations between data_x and data_y
        # rowvar set to false since data_x & data_y don't have the same shapes
        # rowvar=False returns a correlation matrix where each column is compared with every column

        # Step 2: Extract the correlation array for our label column
        correlation_y = correlation_matrix[:-1,-1]
        # Since we only care about the correlation values with data_y, we only want the last column
        # The last column ends in a 1 since data_y will have a correlation of 1 with itself, so remove it
        
        # Step 3: Find the column that has the LARGEST correlation with label
        i = np.argmax(np.abs(correlation_y)) # This approach will automatically choose the first value if all are equal
        # argmax finds the maximum value and returns the position of it, which is what we care about in the end
        # Note, we care about finding the largest value in general, not the largest positive value, so we need to take the absolute value
        return i
    

    def find_SplitVal(self, data_x, i):
        SplitVal = np.median(data_x[:,i])
        return SplitVal
    
    
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
            return np.array([-1, np.mean(data_y), np.nan, np.nan])
        
        # if there is only one value in y (example: y=[1,1,1,1,1,1]), return
        if len(np.unique(data_y)) == 1: return np.array([-1, data_y[0], np.nan, np.nan])
        
        # else: determine best feature i to split on
        i = self.find_best_feature(data_x, data_y)
        
        # Find the Split Value [median]
        SplitVal = self.find_SplitVal(data_x, i)
        LeftSplitCond = data_x[:,i]<=SplitVal
        # On the off chance that every condition in LeftSplitCond is all true or all false, return
        if (np.all((LeftSplitCond == False)) | np.all((LeftSplitCond == True))):
            return np.array([-1, np.mean(data_y), np.nan, np.nan])
        RightSplitCond = data_x[:,i]>SplitVal
        
        # Build Left Side of Tree
        leftdata_x,leftdata_y = data_x[LeftSplitCond], data_y[LeftSplitCond]
        lefttree = self.build_tree(leftdata_x,leftdata_y)
        
        # Build Right Side of Tree
        rightdata_x,rightdata_y = data_x[RightSplitCond], data_y[RightSplitCond]
        righttree = self.build_tree(rightdata_x,rightdata_y) 
        
        # Return Full Tree
        root = np.array([i, SplitVal, 1, lefttree.reshape([-1,4]).shape[0]+1]) 
        # Note: np.append was not working as intended. Used np.concatenate
        tree = np.concatenate([root,lefttree,righttree])
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
                SplitCon = point[i]<=SplitVal
                if SplitCon:
                    node += self.learner[node][2]
                else:
                    node += self.learner[node][3]
                node = int(node)
            ypred = np.append(ypred,self.learner[node][1])
        return ypred