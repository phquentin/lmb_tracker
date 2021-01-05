import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

class GM():
    """
    Gaussian Mixture PDF

    Parameters
    ----------
    params.dim_x : int
        Dimension (number) of state variables
    params.dim_z : int
        Dimension (number) of measurement inputs
    params.P_init : numpy.array(dim_x, dim_x)
        Initial state covariance matrix
    params.F : numpy.array(dim_x, dim_x)
        State Transition matrix
    params.Q : numpy.array(dim_x, dim_x)
        Process Noise matrix
    params.H : numpy.array(dim_z, dim_x)
        Observation model
    params.R : numpy.array(dim_z, dim_z)
        Observation Noise matrix
    params.log_p_detect : float
        Detection probability as log-likelihood
    params.log_q_detect : float
        Inverse of detection probability as log-likelihood
    params.log_kappa : float
        Clutter intensity as log-likelihood
    params.log_w_prun_th
        Log-likelihood threshold of gaussian mixture weight for pruning 
    params.mc_merging_dist_th
        Merging threshold of squared Mahalanobis distance between two mixture components
    x0 : numpy.array(dim_x) optional
        Initial state estimate
    prior_mc : numpy.array (dtype=self.dtype_mc), optional
        Prior gaussian mixture components for initialization

    Attributes
    ----------
    log_w_sum : float
        log of sum of mixture weights
    log_eta_z : float
        log-likelihood of the last measurement association
    mc : numpy.array
        Array of mixture components, each described by its mean, covariance
        and log of normalized mixture weight (weights summing up to 1)
    """
    
    def __init__(self, params, x0=None, prior_mc=None):
        self.params = params
        self.dim_x = self.params.dim_x
        self.F = self.params.F
        self.H = self.params.H
        self.Q = self.params.Q
        self.R = self.params.R
        self.log_p_detect = self.params.log_p_detect
        self.log_q_detect = self.params.log_q_detect
        self.log_kappa = self.params.log_kappa
        self.log_w_prun_th = self.params.log_w_prun_th
        self.mc_merging_dist_th = self.params.mc_merging_dist_th
        # data type of mixture component entry
        self.dtype_mc = np.dtype([('log_w', 'f8'),
                                ('x', 'f4', self.dim_x),
                                ('P', 'f4', (self.dim_x, self.dim_x))])
        if prior_mc is not None and prior_mc.dtype == self.dtype_mc:
            self.mc = prior_mc
        else:
            # init ndarray of mixture components with one Gaussian
            self.mc = np.zeros(1, dtype=self.dtype_mc)
            self.mc[0]['log_w'] = 0.0
            self.mc[0]['x'] = x0 if x0 is not None else np.zeros(self.dim_x)
            self.mc[0]['P'] = self.params.P_init

        self.log_w_sum = logsumexp(self.mc['log_w'])

    def predict(self):
        """
        Predict Gaussian Mixture state components (x, P) 
        
        Each mixture component is predicted using the Kalman equations
            x = Fx
            P = FPF' + Q
        """
        # x and P of all mixture components are predicted with array broadcasting
        # In their original form, shapes of x (mc['x']) and F are not aligned,
        # thus x has to be transposed.
        # F: (dim_x, dim_x)
        # x: (len(mc), dim_x)
        # -> x = (Fx')' -> x = xF'
        self.mc['x'] = np.dot(self.mc['x'], self.F.T)
        # Iterate through all mixture components
        for i in range(len(self.mc)):
            self.mc[i]['P'] = np.dot(np.dot(self.F, self.mc[i]['P']), self.F.T)
        # Add Process Noise matrix via broadcasting
        self.mc['P'] = self.mc['P'] + self.Q

    def correct(self, z):
        """
        Update PDF with a new measurement

        Corrects each mixture component and their weights 
        with the new measurement using the Kalman filter
            x = x + Ky
            P = P - KSK'
            with
                y = z - Hx
                S = HPH' + R
                K = PH'inv(S)

        Additionally, the log-likelihood log_eta_z of this measurement association
        is computed according to equation for calculating eta_z:
        log_eta_z = ln(eta_z) = ln( (p_detect / kappa) * sum( w_cmpnts * N(z; z_+, S) ) )
        with N(z; z_+, S) being the Gaussian measurement likelihood.

        If no measurement (None or empty array) is given, it is treated 
        as a missed detection. Therefore, the mixture components are not 
        updated and the log_likelihood of the association is updated with
        the probability for a missed detection.

        Parameters
        ----------
        z: array_like
            A new measurement input (one target)
        """
        if z is None or len(z) == 0:
            self.log_eta_z = self.log_w_sum + self.log_q_detect
        else:
            # TODO The computation can be optimized by using numba and guvectorize.
            # This enables the efficient use of broadcasting, such that the current loop
            # over all mixture components can be replaced by one sequential computation.
            for i, cmpnt in enumerate(self.mc):
                y = z - np.dot(self.H, cmpnt['x'])
                S = np.dot(self.H, np.dot(cmpnt['P'], self.H.T)) + self.R
                S_inv = np.linalg.inv(S)
                K = np.dot(cmpnt['P'], np.dot(self.H.T, S_inv))
                self.mc[i]['x'] = cmpnt['x'] + np.dot(K, y)
                self.mc[i]['P'] = cmpnt['P'] - np.dot(np.dot(K, S), K.T)
                # Scipy computation for numerical stability. The self-implemented version
                # is more computational efficient by reusing precomputed values.
                self.mc[i]['log_w'] = cmpnt['log_w'] + multivariate_normal.logpdf(np.dot(self.H, cmpnt['x']), mean=z, cov=S)
                #self.mc[i]['log_w'] = cmpnt['log_w'] - 0.5 * (len(y) * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + np.dot(y.T, np.dot(S_inv, y)))

            # Normalization of mixture weights to receive updated weights
            self.log_w_sum = logsumexp(self.mc['log_w'])
            self.mc['log_w'] -= self.log_w_sum
            # Computation of log_likelihood of this target-measurement association
            self.log_eta_z = self.log_w_sum + self.log_p_detect - self.log_kappa

        return self


    def overwrite_with_merged_pdf(self, pdfs, log_hyp_weights):
        """
        Merge multiple Gaussian mixtures into one mixture and overwrite current mixture

        Parameters
        ----------
        pdfs : array_like
            Array/list of Gaussian mixture PDFs objects
        log_hyp_weights : array_like
            Normalized hypothesis weights of the PDFs in log representation
        """
        
        # Normalize mixture component weights
        mcs = [pdf.mc for pdf in pdfs]
        for log_hyp_weight, mc in zip(log_hyp_weights, mcs):
            mc['log_w'] += log_hyp_weight

        self.mc = np.concatenate(mcs)


    def reduce_pdf(self):
        """
        Reduce the complexity of the PDF by pruning and merging
        """
        self._prune_gaussians()
        self._merge_gaussians()


    def _merge_gaussians(self):
        """
        Merge Gaussian Mixture components which are closer than a defined threshold

        All mixture components are compared with eachother. Components with a 
        squared Mahalanobis distance below a specified value are combined
        into one mixture component by averaging their mean and covariance weighed
        by their component weights.
        
        The algorithm uses an acceptance mask on the existing mixture components 
        to avoid creating new component objects.
        """
        num_cmpnts = len(self.mc)
        mc_ids = np.arange(num_cmpnts)

        accept_mask = np.zeros(num_cmpnts, dtype=bool) 
        process_mask = np.ones(num_cmpnts, dtype=bool) 

        sort_ids = np.argsort(self.mc['log_w'])[::-1]  
        
        for cmpnt_id in sort_ids:
            if process_mask[cmpnt_id]:
                if np.count_nonzero(process_mask) > 1:
                    P_inv = np.linalg.inv(self.mc[cmpnt_id]['P'])
                    # init merge mask with False
                    merge_mask = np.zeros(len(self.mc[process_mask]), dtype=bool)
                    # Compare current component with all unprocessed components
                    for i, cmpnt in enumerate(self.mc[process_mask]):
                        diff = self.mc[cmpnt_id]['x'] - cmpnt['x']
                        # Squared Mahalanobis distance
                        dist = np.dot(diff, np.dot(P_inv, diff))
                        if dist < self.mc_merging_dist_th: 
                            merge_mask[i] = True

                    # Merging just with itself is skipped
                    if np.count_nonzero(merge_mask) > 1: 
                        merge_ids = mc_ids[process_mask][merge_mask]

                        # Merge Gaussian mixture components by averaging
                        weights = np.exp(self.mc[merge_ids]['log_w'])
                        x_new = np.average(self.mc[merge_ids]['x'], weights=weights, axis=0)
                        P_new = np.average(self.mc[merge_ids]['P'], weights=weights, axis=0)
                        self.mc[cmpnt_id]['log_w'] = np.log(sum(weights))
                        self.mc[cmpnt_id]['x'] = x_new
                        self.mc[cmpnt_id]['P'] = P_new

                        process_mask[merge_ids] = False
                
                process_mask[cmpnt_id] = False
                accept_mask[cmpnt_id] = True

        self.mc = self.mc[accept_mask]


    def _prune_gaussians(self):
        '''
        Prune gaussian mixture components whose weights are lower than the defined threshold self.log_w_prun_th
        '''

        # Pruning
        prune_mask = self.mc['log_w'] > self.log_w_prun_th
        self.mc = self.mc[prune_mask] 

        # Normalize after pruning. This may not be necessary depending on the extraction logic; relative weights may be okay 
        self.log_w_sum = logsumexp(self.mc['log_w'])
        self.mc['log_w'] -= self.log_w_sum

        