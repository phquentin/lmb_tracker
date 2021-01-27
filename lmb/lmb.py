from operator import attrgetter
import numpy as np

from .parameters import TrackerParameters
from .target import Target
from .gm import GM
from .utils import esf

class LMB():
    """
    Main class of the labeled multi bernoulli filter implementation.

    Parameters
    ----------
    params : TrackerParameter, optional
        Parameter object containing all tracker parameters required by all subclasses.
        Gets initialized with default parameters, in case no object is passed.
    
    LMB-Class specific Parameters
    ------
    params.n_targets_max : int
        Maximum number of targets
    params.log_p_survival : float
        Target survival probability as log-likelihood
    params.p_birth : float
        Maximum birth probability of targets
    params.adaptive_birth_th : float
        Birth probability threshold for existence probability of targets
    params.log_r_prun_th : float
        Log-likelihood threshold of target existence probability for pruning
    self.params.log_r_sel_th : float
        Log-likelihood threshold of target existence probability for selection
    self.params.num_assignments : int
        Maximum number of hypothetical assignments created by the ranked assignment
    dtype_extract : numpy dtype
        Dtype of the extracted targets

    Attributes
    ----------
    targets : array_like
        List of currently active targets
    """
    def __init__(self, params=None):
        self._ts = 0
        self.params = params if params else TrackerParameters()
        self.n_targets_max = self.params.n_targets_max
        self.log_p_survival = np.log(self.params.p_survival)
        self.p_birth = self.params.p_birth
        self.adaptive_birth_th = self.params.adaptive_birth_th
        self.log_r_prun_th = self.params.log_r_prun_th
        self.log_r_sel_th = self.params.log_r_sel_th
        self.ranked_assign = self.params.ranked_assign
        self.ranked_assign_num = self.params.num_assignments
        self.dtype_extract = np.dtype([('x', 'f4', self.params.dim_x),
                                       ('P', 'f4', (self.params.dim_x, self.params.dim_x)),
                                       ('r','f4'),
                                       ('label', 'f4'),
                                       ('ts','f4')])
                        
        self.targets = [] # list of currently tracked targets


    def update(self,z):
        """
        Main function to update the internal tracker state with a measuerement

        Parameters
        ----------
        z: measurement object (class to be implemented)

        Returns
        -------
        out: ndarray
            updated and extracted targets of the format : np.dtype([('x', 'f4', dim_x),
                                                                    ('P', 'f4', (dim_x, dim_x)),
                                                                    ('r','f4'),
                                                                    ('label', 'f4'),
                                                                    ('ts','f4')])
        """
        
        self._ts += 1
        print('Update step ', self._ts)

        self.predict()
        self.correct(z)

        return self.extract(self._select())

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
        if N > 0:
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
            self.ranked_assign(C, hyp_weights, self.ranked_assign_num)
            ## 3. Calculate resulting existence probability of each target
            for i, target in enumerate(self.targets):
                target.correct(hyp_weights[i,:-1])

        else:
            # No tracked targets yet, set all assignment weights to 0
            hyp_weights = np.zeros((1, M + 2))

        self._adaptive_birth(z, hyp_weights[:,:-2])
        ## 4. Prune targets
        self._prune()
        print('Corrected, born, and pruned targets ', self.targets)


    def _adaptive_birth(self, Z, assign_weights):
        """
        Adaptive birth of targets based on measurement

        New targets are born at the measurement locations based on the 
        assignment probabilities of these measurements: The higher the 
        probability of a measurement being assigned to any existing target,
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
                # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
                # This results in setting the birth prob to (1 - prob_assign).
                self.lambda_b = not_assigned_sum
                # limit the birth existence probability to the configured p_birth
                prob_birth = np.minimum(self.p_birth, self.lambda_b * (1 - prob)/not_assigned_sum)
                ## Debug output:
                # print("sum ", not_assigned_sum, " assign prob ", prob, " prob_birth ", prob_birth)
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


    def _select(self):
        """
        Select the most likely targets

        Compute the most likely cardinality (number) of targets and 
        select the corresponding number of targets with the highest existence probability.
        The cardinality is obtained by computing the cardinality distribution of
        the multi-Bernoulli RFS and selecting the cardinality with highest probability.

        TODO: the mean cardinality of a multi-Bernulli RFS can be obtained by sum(r).
        The applicability for the selection step should be investigated. 

        Returns
        -------
        out: list
            selected targets
        """
        # exclude the targets born in this update step
        selected_targets = [t for t in self.targets if t.label.split('.')[0] != str(self._ts)]
        # get the existence probabilities of the targets
        r = [np.exp(t.log_r) for t in selected_targets]
        r = np.asarray(r)
        # limit the existence probabilities for numerical stability (avoiding divide by 0)
        r = np.minimum(r, 1. - 1e-9)
        r = np.maximum(r, 1e-9)
        # Calculate the cardinality distribution of the multi-Bernoulli RFS
        cdn_dist = np.prod(1.0 - r) * esf(r / (1. - r))
        est_cdn = np.argmax(cdn_dist)
        num_tracks = min(est_cdn, self.n_targets_max)

        selected_targets.sort(key=attrgetter('log_r'), reverse=True)
        selected_targets = selected_targets[:num_tracks]

        print("Selected targets: ", selected_targets)

        return selected_targets


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


    def extract(self, selected_targets):
        """
        Extract selected targets from the LMB class instance 

        Extract the selected targets with their labels, existence probabilities and their states x
        and covariances P of their corresponding most likely gaussian mixture component.

        Parameters
        ----------
        selected_targets : list
            List of class Targets instances

        Returns
        -------
        out: ndarry
            Ndarray of dtype: self.dtype_extract
        """

        extracted_targets = np.zeros(len(selected_targets), dtype=self.dtype_extract)   

        for i, target in enumerate(selected_targets):
            mc_extract_ind = np.argmax(target.pdf.mc['log_w'])
            extracted_targets[i]['x'] = target.pdf.mc[mc_extract_ind]['x']
            extracted_targets[i]['P'] = target.pdf.mc[mc_extract_ind]['P']
            extracted_targets[i]['r'] = np.exp(target.log_r)
            extracted_targets[i]['label'] = target.label
            extracted_targets[i]['ts'] = self._ts
           
        return extracted_targets      
      
