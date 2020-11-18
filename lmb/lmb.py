import numpy as np

from .parameters import TrackerParameters
from .target import Target
from .gm import GM

class LMB():
    """
    Main class of the labeled multi bernoulli filter implementation.

    Parameters
    ----------
    params : TrackerParameter, optional
        Parameter object containing all tracker parameters required by all subclasses.
        Gets initialized with default parameters, in case no object is passed.
    
    LMB-Class specific Paramters
    ------
    params.log_p_survival : float
        Target survival probability as log-likelihood

    Attributes
    ----------
    targets : array_like
        List of currently active targets
    """
    def __init__(self, params=None):
        self.params = params if params else TrackerParameters()
        self.log_p_survival = np.log(self.params.p_survival)
        self.ranked_assign = self.params.ranked_assign

        self.targets = [] # list of currently tracked targets
        self.targets.append(Target("0", pdf=GM(params=params)))
        self.targets.append(Target("1", pdf=GM(params=params)))

    def update(self,z):
        """
        Main function to update the internal tracker state with a measuerement

        Parameters
        ----------
        z: measurement object (class to be implemented)

        Returns
        -------
        out: array_like
            updated tracks
        """
        self.predict()
        self.correct(z)

        return self.select()

    def predict(self):
        """
        Prediction step of LMB tracker

        Predicts states and existence probabilities of every currently tracked target
        """
        for target in self.targets:
            target.predict(self.log_p_survival)

    def correct(self, z):
        """
        Correction step of LMB tracker

        Correct the predicted track states and their existence probabililities using the new measurement

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        ## 1. Create target-measurement associations and calculate each log_likelihood
        ## create association cost matrix with first column for death and second for missed detection
        ## (initialize with min prob --> high negative value due to log): 
        N = len(self.targets)
        M = len(z['z'])
        C = np.zeros((N, 2 + M))
        ## compute entries of cost matrix for each target-measurement association (including misdetection)
        for i, target in enumerate(self.targets):
            # missed detection (column 1) and associations
            C[i, range(M + 1)] = target.create_assignments(z)
            # died or not born
            C[i, (M + 1)] = target.nll_false()
        print('C \n',C)
        ## Ranked assignment 
        ## 2. Compute hypothesis weights using specified ranked assignment algorithm
        hyp_weights = -500 * np.ones((N, M + 1))
        self.ranked_assign(C, hyp_weights)
        print('hyp_weights \n', hyp_weights)
        ## 3. Calculate resulting existence probability of each target
        for i, target in enumerate(self.targets):
            target.correct(hyp_weights[i, ])
        ## 4. Prune targets
        self._prune()

    def _prune(self):
        """
        Pruning of tracks

        Selection according to the configured pruning threshold for the existence probability.
        Afterwards, limit the remaining tracks to the configured maximum number based on 
        descending existence probability. 
        """
        pass

    def select(self):
        """
        Select tracks based on existence probability

        Computes the most likely number of tracks and selects for this number of tracks the Gaussian
        mixture component with the highest weight of the tracks with the highest existence probability.
        """
        pass

    def _spawn(self):
        """
        Spawn new target instances
        """
        pass
