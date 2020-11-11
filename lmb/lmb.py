import numpy as np

from .parameters import Parameters
from .target import Target
from .gm import GM

class LMB():
    """
    Main class of the labeled multi bernoulli filter implementation.
    """
    def __init__(self, params=None):
        self.params = params if params else Parameters()
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
        ## 2. Compute hypothesis weights and resulting existence probability of each target
        # for target in self.targets:
        #    target.correct()
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
