import numpy as np
from scipy.special import logsumexp

class GM():
    """
    Gaussian Mixture PDF

    Parameters
    ----------
    x0 : numpy.array(dim_x) optional
        Initial state estimate
    dim_x : int
        Dimension (number) of state variables
    dim_z : int
        Dimension (number) of measurement inputs
    P_init : numpy.array(dim_x, dim_x)
        Initial state covariance matrix
    F : numpy.array(dim_x, dim_x)
        State Transition matrix
    Q : numpy.array(dim_x, dim_x)
        Process Noise matrix
    H : numpy.array(dim_z, dim_x)
        Observation model
    R : numpy.array(dim_z, dim_z)
        Observation Noise matrix

    Attributes
    ----------
    log_w_sum : float
        log of sum of mixture weights
    mc : numpy.array
        Array of mixture components, each described by its mean, covariance
        and log of normalized mixture weight (weights summing up to 1)
    """
    
    def __init__(self, params, x0=None):
        self.params = params
        self.dim_x = self.params.dim_x
        self.F = self.params.F
        self.H = self.params.H
        self.Q = self.params.Q
        self.R = self.params.R
        # data type of mixture component entry
        self.dtype_mc = np.dtype([('log_w', 'f8'),
                                ('x', 'f4', self.dim_x),
                                ('P', 'f4', (self.dim_x, self.dim_x))])
        # init list of mixture components with one Gaussian
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

        Parameters
        ----------
        z: array_like
            A new measurement input (one target)
        """
        # @todo The computation can be optimized by using numba and guvectorize.
        # This enables the efficient use of broadcasting, such that the current loop
        # over all mixture components can be replaced by one sequential computation.
        for i, cmpnt in enumerate(self.mc):
            y = z - np.dot(self.H, cmpnt['x'])
            S = np.dot(self.H, np.dot(cmpnt['P'], self.H.T)) + self.R
            S_inv = np.linalg.inv(S)
            K = np.dot(cmpnt['P'], np.dot(self.H.T, S_inv))
            self.mc[i]['x'] = cmpnt['x'] + np.dot(K, y)
            self.mc[i]['P'] = cmpnt['P'] - np.dot(np.dot(K, S), K.T)
            self.mc[i]['log_w'] = cmpnt['log_w'] - 0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + np.dot(y, np.dot(S_inv, y)))

        # Normalization of mixture weights
        self.log_w_sum = logsumexp(self.mc['log_w'])
        self.mc['log_w'] -= self.log_w_sum

    def merge(self):
        """
        Merge Gaussian Mixture components which are closer than a defined threshold
        """
        pass
