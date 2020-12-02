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
    
    LMB-Class specific Parameters
    ------
    params.log_p_survival : float
        Target survival probability as log-likelihood

    dtype_extract : numpy dtype
        dtype of the extracted targets

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
        self.targets.append(Target(1, pdf=GM(params=params)))
        self.targets.append(Target(2, pdf=GM(params=params, x0=[20.,50.,0.,0.])))
        self.targets.append(Target(3, pdf=GM(params=params, x0=[1.,-1.,0.,0.])))
        self.targets.append(Target(4, pdf=GM(params=params, x0=[-10.,-10.,0.,0.])))
        self.targets.append(Target(5, pdf=GM(params=params, x0=[10.,10.,0.,0.])))

        self.dtype_exctract = np.dtype([('x', 'f4', self.params.dim_x),
                                        ('P', 'f4', (self.params.dim_x, self.params.dim_x)),
                                        ('r','f4'),
                                        ('label', 'u4')])
                        

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

        return self.extract(self._select())

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
            # associations and missed detection (second-last column)
            C[i, range(M + 1)] = target.create_assignments(z)
            # died or not born (last column)
            C[i, (M + 1)] = target.nll_false()

        ## Ranked assignment 
        ## 2. Compute hypothesis weights using specified ranked assignment algorithm
        hyp_weights = np.zeros((N, M + 2))
        self.ranked_assign(C, hyp_weights)
        hyp_weights = np.log(hyp_weights)
        ## 3. Calculate resulting existence probability of each target
        for i, target in enumerate(self.targets):
            target.correct(hyp_weights[i,:-1])
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

    def _select(self):
        """
        Select tracks based on existence probability

        Computes the most likely number of tracks and selects for this number of tracks the Gaussian
        mixture component with the highest weight of the tracks with the highest existence probability.
        """

        #select_mask = [target.log_r > self.params.sel_log_r for target in self.targets]
        #(d for d, s in izip(data, selectors) if s)
        selected_targets = [target for target in self.targets if target.log_r > self.params.sel_log_r]
        print(len(selected_targets))
        
        return selected_targets

    def _spawn(self):
        """
        Spawn new target instances
        """
        pass

    def extract(self, selection):

        extracted_tracks = np.zeros(len(selection), dtype=self.dtype_exctract)   

        for i, target in enumerate(selection):
            mc_extract_ind = np.argmax(target.pdf.mc['log_w'])
            extracted_tracks[i]['x'] = target.pdf.mc[mc_extract_ind]['x']
            extracted_tracks[i]['P'] = target.pdf.mc[mc_extract_ind]['P']
            extracted_tracks[i]['r'] = target.log_r
            extracted_tracks[i]['label'] = target.label
           
        return extracted_tracks      
      

        
