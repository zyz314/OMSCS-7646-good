import numpy as np
from BagLearner import BagLearner
from LinRegLearner import LinRegLearner
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose=verbose
        self.insanity = np.array([BagLearner(LinRegLearner) for _ in range(20)])
    def __repr__(self): return(f"InsaneLearner(learner = {self.insanity}, verbose = {self.verbose})")
    def __str__(self): return(f"InsaneLearner(learner = {self.insanity}, verbose = {self.verbose})")
    def add_evidence(self,data_x, data_y):
        for learner in self.insanity: learner.add_evidence(data_x,data_y)
    def author(self): return "mshihab6"
    def query(self, points):
        predictions = []
        for learner in self.insanity: predictions.append(learner.query(points))
        return np.mean(np.array(predictions),axis=0)