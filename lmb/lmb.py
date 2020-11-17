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
        self.targets = [] # list of currently tracked targets
        self.targets.append(Target("0", pdf=GM(params=params)))

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
        M = len(z)
        C = np.zeros((N, 2 + M))
        ## compute entries of cost matrix for each target-measurement association (including misdetection)
        for i, target in enumerate(self.targets):
            # missed detection (column 1) and associations
            C[i, 1:] = target.create_associations(z)
            # died or not born
            C[i, 0] = target.nll_false()

        ## Ranked assignment using Gibbs sampler
        ## 2. Compute hypothesis weights using Gibbs sampler
        ## 3. Calculate resulting existence probability of each target
        # for target in self.targets:
        #    target.correct(assignment_weights)
        ## 3. Prune targets
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
