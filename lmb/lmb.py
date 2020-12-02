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
    params.p_birth : float
        Maximum birth probability of targets
    params.adaptive_birth_th : float
        Birth probability threshold for existence probability of targets
    params.log_r_prun_th : float
        Log-likelihood threshold of target existence probability for pruning

    Attributes
    ----------
    targets : array_like
        List of currently active targets
    """
    def __init__(self, params=None):
        self._ts = 0
        self.params = params if params else TrackerParameters()
        self.log_p_survival = np.log(self.params.p_survival)
        self.p_birth = self.params.p_birth
        self.adaptive_birth_th = self.params.adaptive_birth_th
        self.log_r_prun_th = self.params.log_r_prun_th
        self.ranked_assign = self.params.ranked_assign

        self.targets = [] # list of currently tracked targets
        self._spawn_target(log_r=0., x0=None)
        self._spawn_target(log_r=0., x0=[20.,50.,0.,0.])
        self._spawn_target(log_r=0., x0=[1.,-1.,0.,0.])
        self._spawn_target(log_r=0., x0=[-10.,-10.,0.,0.])
        self._spawn_target(log_r=0., x0=[10.,10.,0.,0.])

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
        self._ts += 1
        print('Update step ', self._ts)

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
        
        print('Predicted targets ', self.targets)


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
            # associations and missed detection (second-last column)
            C[i, range(M + 1)] = target.create_assignments(z)
            # died or not born (last column)
            C[i, (M + 1)] = target.nll_false()

        ## Ranked assignment 
        ## 2. Compute hypothesis weights using specified ranked assignment algorithm
        hyp_weights = np.zeros((N, M + 2))
        self.ranked_assign(C, hyp_weights)
        #hyp_weights = np.log(hyp_weights)
        ## 3. Calculate resulting existence probability of each target
        for i, target in enumerate(self.targets):
            target.correct(hyp_weights[i,:-1])

        print('Corrected targets ', self.targets)
        self._adaptive_birth(z, hyp_weights[:,:-2])
        ## 4. Prune targets
        self._prune()
        print('Corrected, born, and pruned targets ', self.targets)


    def _adaptive_birth(self, Z, assign_weights):
        """
        Adaptive birth of targets based on measurement

        New targets are born at the measurement locations based on the 
        assignment probabilities of these measurements: The higher the 
        probability of a measurement being assign to any existing target,
        the lower the birth probability of a new target at this position.

        The implementation is based on the proposed algorithm in
        S. Reuter et al., "The Labeled Multi-Bernoulli Filter", 2014

        Parameters
        ----------
        Z : array_like
            measurements
        assign_weights : array_like
            Weights of all track-measurement assignments (without missed detection or deaths)
            Shape: num_tracks x num_measurements
        """
        # Probability of each measurement being assigned to an existing target
        z_assign_prob = np.sum(assign_weights, axis=0)
        not_assigned_sum = sum(1 - z_assign_prob)

        if not_assigned_sum > 1e-9:
            for z, prob in zip(Z, z_assign_prob):
                # limit the birth existence probability to the configured p_birth
                prob_birth = np.minimum(self.p_birth, (1 - prob)/not_assigned_sum)
                # Spawn only new targets which exceed the existence prob threshold
                if prob_birth > self.adaptive_birth_th:
                    self._spawn_target(np.log(prob_birth), x0=[z['z'][0], z['z'][1], 0., 0.])


    def _prune(self):
        """
        Pruning of tracks

        Selection according to the configured pruning threshold for the existence probability.
        TODO: limit the remaining tracks to the configured maximum number based on 
        descending existence probability. 
        """
        self.targets = [t for t in self.targets if t.log_r > self.log_r_prun_th]


    def select(self):
        """
        Select tracks based on existence probability

        Computes the most likely number of tracks and selects for this number of tracks the Gaussian
        mixture component with the highest weight of the tracks with the highest existence probability.
        """
        pass

    def _spawn_target(self, log_r, x0):
        """
        Spawn new target instances

        Parameters
        ----------
        log_r : float
            Log likelihood of initial existence probability
        x0 : array_like
            Initial state
        """
        label = '{}.{}'.format(self._ts, len(self.targets)) 
        self.targets.append(Target(label, log_r=log_r, pdf=GM(params=self.params, x0=x0)))
